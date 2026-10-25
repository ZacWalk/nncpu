//! Model loading: hyperparameters, tensors and their load-time decode.

use super::{Arch, Hparams, Layer, MAX_EXPERTS, ModelData, Pooling, WeightFormat};
use crate::error::{Result, bail, err};
use crate::gguf::{GgufFile, GgufValue, ggml_type_name};
use crate::kernels::{self, RopeMode};
use crate::pool::{self, SyncPtr};
use crate::runtime::{AlignedBuf, DType, Tensor, f32_to_bf16};
use crate::tokenizer::Vocab;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Metadata helpers.
// ---------------------------------------------------------------------------

fn value_int(v: &GgufValue) -> Option<i32> {
    match *v {
        GgufValue::U8(x) => Some(x as i32),
        GgufValue::U16(x) => Some(x as i32),
        GgufValue::U32(x) => i32::try_from(x).ok(),
        GgufValue::U64(x) => i32::try_from(x).ok(),
        GgufValue::I8(x) => Some(x as i32),
        GgufValue::I16(x) => Some(x as i32),
        GgufValue::I32(x) => Some(x),
        GgufValue::I64(x) => i32::try_from(x).ok(),
        _ => None,
    }
}

/// An integer hparam: a scalar, or the first element of a per-layer array.
pub(super) fn kv_int(f: &GgufFile, key: &str) -> Option<i32> {
    match f.get(key)? {
        GgufValue::Array(_, a) => value_int(a.first()?),
        v => value_int(v),
    }
}

/// A per-layer integer hparam: element `li` of an array, or the scalar.
fn kv_int_at(f: &GgufFile, key: &str, li: usize) -> Option<i32> {
    match f.get(key)? {
        GgufValue::Array(_, a) => value_int(a.get(li)?),
        v => value_int(v),
    }
}

/// A positive integer hparam.
fn kv_pos(f: &GgufFile, key: &str) -> Option<usize> {
    kv_int(f, key).filter(|&v| v > 0).map(|v| v as usize)
}

fn kv_float(f: &GgufFile, key: &str) -> Option<f32> {
    f.get(key)?.as_float().map(|v| v as f32)
}

fn kv_bool(f: &GgufFile, key: &str) -> Option<bool> {
    f.get(key)?.as_bool()
}

/// A per-layer flag: element `li` of an array of bools (or integers).
fn kv_flag_at(f: &GgufFile, key: &str, li: usize) -> Option<bool> {
    let GgufValue::Array(_, a) = f.get(key)? else {
        return None;
    };
    let e = a.get(li)?;
    match *e {
        GgufValue::Bool(b) => Some(b),
        _ => value_int(e).map(|i| i != 0),
    }
}

/// Byte size of `n` weights in `fmt`, secondary arrays included.
fn fmt_bytes(fmt: WeightFormat, n: usize) -> usize {
    match fmt {
        WeightFormat::Q8_0 => n + n / 32 * 2,
        WeightFormat::Q4 => n / 2 + n / 32 * 4,
        WeightFormat::Bf16 => n * 2,
    }
}

fn fmt_type(fmt: WeightFormat) -> DType {
    match fmt {
        WeightFormat::Q8_0 => DType::Q8_0,
        WeightFormat::Q4 => DType::Q4S,
        WeightFormat::Bf16 => DType::BF16,
    }
}

/// Elements per storage block of a type dequantize_to_f32 can read, or 0.
fn decode_block_elems(ggml_type: u32) -> u32 {
    match ggml_type {
        0 | 1 | 30 => 1, // F32, F16, BF16
        t => kernels::quant_block_elems(t),
    }
}

// ---------------------------------------------------------------------------
// Load progress, measured in bytes of the file processed.
// ---------------------------------------------------------------------------

/// Load progress in [0, 1]; return false to cancel the load.
pub(crate) type ProgressFn<'a> = &'a mut dyn FnMut(f32) -> bool;

struct Progress<'a> {
    f: Option<ProgressFn<'a>>,
    total: u64,
    done: u64,
    reported: f32,
    cancelled: bool,
}

impl Progress<'_> {
    /// Returns false once the callback has asked to cancel.
    fn advance(&mut self, bytes: u64) -> bool {
        self.done += bytes;
        let Some(f) = self.f.as_mut() else { return true };
        if self.cancelled {
            return false;
        }
        let p = if self.total > 0 {
            self.done as f64 / self.total as f64
        } else {
            1.0
        };
        let pf = p.min(1.0) as f32;
        if pf - self.reported < 0.005 {
            return true;
        }
        self.reported = pf;
        if !f(pf) {
            self.cancelled = true;
        }
        !self.cancelled
    }
}

// ---------------------------------------------------------------------------
// Tensor construction.
//
// Every tensor is loaded for a role. A `Matrix` (anything a gemv or an
// embedding lookup reads) always ends up BF16, Q8_0 or Q4 — whatever the file
// stores — because those are the layouts the kernels take; so does each
// matrix of an `Experts` stack. A `Vector` (norm gammas, scales, RoPE
// factors, and the small MoE router, whose ranking quantization could upset)
// always ends up F32. Anything already in its target layout is used in place
// from the mapping; everything else is decoded once, here, in parallel.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Role {
    Matrix,
    Vector,
    /// [cols, rows, n_expert]: one split-layout matrix per expert, back to back.
    Experts,
}

/// 16384 elements: a multiple of every block size in play (32 and 256), so no
/// source block straddles a chunk, and small enough that each thread's F32
/// staging buffer stays in L2.
const CHUNK: usize = 256 * 64;

struct Decode<'a> {
    ggml_type: u32,
    src: &'a [u8],
    block_bytes: usize,
    block_elems: usize,
    /// Weights in this matrix.
    n: usize,
    /// Vector role: write F32 straight into dst.
    to_f32: bool,
    /// Matrix role: encode into this.
    fmt: WeightFormat,
}

/// Encode `src` (F32 weights at flat offset `off`) into `dst`, which holds
/// all `n` weights of a matrix in `fmt`'s split layout.
///
/// # Safety
/// `dst` covers fmt_bytes(fmt, n) bytes, 64-byte aligned, and no other thread
/// writes the parts of it that weights [off, off + src.len()) map to.
unsafe fn encode_chunk(fmt: WeightFormat, dst: SyncPtr<u8>, n: usize, src: &[f32], off: usize) {
    let take = src.len();
    unsafe {
        match fmt {
            WeightFormat::Q8_0 => {
                let qs = SyncPtr(dst.0 as *mut i8).slice_mut(off, take);
                let scales = SyncPtr(dst.0.add(n) as *mut u16).slice_mut(off / 32, take / 32);
                kernels::quantize_f32_to_q8_0(src, qs, scales);
            }
            WeightFormat::Q4 => {
                let qs = dst.slice_mut(off / 2, take / 2);
                let scales = dst.0.add(n / 2) as *mut u16;
                let s = SyncPtr(scales).slice_mut(off / 32, take / 32);
                let b = SyncPtr(scales.add(n / 32)).slice_mut(off / 32, take / 32);
                kernels::quantize_f32_to_q4(src, qs, s, b);
            }
            WeightFormat::Bf16 => {
                let bf = SyncPtr(dst.0 as *mut u16).slice_mut(off, take);
                for (o, &v) in bf.iter_mut().zip(src) {
                    *o = f32_to_bf16(v);
                }
            }
        }
    }
}

/// Decode one matrix into `dst` (fmt_bytes(fmt, n) bytes, or n F32s), in
/// chunks shared across the pool. False if the type cannot be decoded.
fn decode(job: &Decode, dst: &mut [u8]) -> bool {
    let n_chunks = job.n.div_ceil(CHUNK);
    let failed = AtomicBool::new(false);
    let out = SyncPtr::new(dst);
    pool::parallel_for(n_chunks as u32, |c0, c1| {
        if c0 >= c1 {
            return;
        }
        let mut tmp = if job.to_f32 { Vec::new() } else { vec![0.0f32; CHUNK] };
        for c in c0 as usize..c1 as usize {
            let off = c * CHUNK;
            let take = CHUNK.min(job.n - off);
            let s0 = off / job.block_elems * job.block_bytes;
            let s1 = (off + take) / job.block_elems * job.block_bytes;
            let src = &job.src[s0..s1];
            if job.to_f32 {
                // SAFETY: chunk c alone writes floats [off, off + take).
                let o = unsafe { SyncPtr(out.0 as *mut f32).slice_mut(off, take) };
                if !kernels::dequantize_to_f32(job.ggml_type, src, o) {
                    failed.store(true, Ordering::Relaxed);
                    return;
                }
            } else {
                if !kernels::dequantize_to_f32(job.ggml_type, src, &mut tmp[..take]) {
                    failed.store(true, Ordering::Relaxed);
                    return;
                }
                // SAFETY: chunk c alone encodes weights [off, off + take).
                unsafe { encode_chunk(job.fmt, out, job.n, &tmp[..take], off) };
            }
        }
    });
    !failed.load(Ordering::Relaxed)
}

struct Loader<'f, 'p> {
    f: &'f GgufFile,
    fmt: WeightFormat,
    prog: Progress<'p>,
    /// Every tensor a load consumes is marked, so that a Qwen file holding
    /// tensors nncpu would silently ignore can be refused.
    used: Vec<bool>,
    buffers: Vec<AlignedBuf>,
    decoded_bytes: usize,
}

impl Loader<'_, '_> {
    /// Load tensor `name` for `role`; Ok(None) when the file lacks it (the
    /// caller decides which are required).
    fn load(&mut self, name: &str, role: Role) -> Result<Option<Tensor>> {
        let Some(idx) = self.f.find_tensor(name) else {
            return Ok(None);
        };
        let gt = &self.f.tensors[idx];
        let n = gt.nelements();
        let src_bytes = gt.nbytes();
        let be = decode_block_elems(gt.ggml_type) as u64;
        if be == 0 || src_bytes == 0 {
            bail!(
                UnsupportedModel,
                "tensor '{name}' is {}, which nncpu cannot decode (it reads F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, \
                 Q8_0, Q4_K, Q5_K and Q6_K)",
                ggml_type_name(gt.ggml_type)
            );
        }
        if n == 0 || !n.is_multiple_of(be) {
            bail!(
                BadFormat,
                "tensor '{name}' has {n} elements, not whole {be}-element blocks"
            );
        }
        let src = self.f.tensor_data(idx).ok_or_else(|| {
            err!(
                BadFormat,
                "tensor '{name}' lies outside the file (is the download incomplete?)"
            )
        })?;

        // An expert stack is decoded as n_mats independent matrices, each in
        // its own complete split layout, so one expert is a plain matrix at
        // an offset (expert_matrix) that every gemv path takes unchanged.
        let stacked = role == Role::Experts;
        if stacked && (gt.n_dims != 3 || gt.ne[2] == 0) {
            bail!(BadFormat, "tensor '{name}' is not a stack of expert matrices");
        }
        let n_mats = if stacked { gt.ne[2] } else { 1 };
        let n_per = n / n_mats;
        if !n_per.is_multiple_of(be) {
            bail!(BadFormat, "tensor '{name}' does not divide into whole-block matrices");
        }

        let mut fmt = self.fmt;
        if role != Role::Vector && fmt != WeightFormat::Bf16 && !gt.ne[0].is_multiple_of(32) {
            fmt = WeightFormat::Bf16; // the split layouts need whole 32-element blocks per row
        }
        let dtype = if role == Role::Vector {
            DType::F32
        } else {
            fmt_type(fmt)
        };
        let ne: Vec<usize> = gt.ne[..gt.n_dims as usize].iter().map(|&d| d as usize).collect();

        // In place when the file already holds the target layout (and the
        // data is aligned for it, as GGUF's alignment guarantees).
        let in_place = if role == Role::Vector {
            gt.ggml_type == 0 && src.as_ptr().cast::<f32>().is_aligned()
        } else {
            fmt == WeightFormat::Bf16 && gt.ggml_type == 30 && src.as_ptr().cast::<u16>().is_aligned()
        };
        let data = if in_place {
            src.as_ptr()
        } else {
            let n_per = n_per as usize;
            let per_mat = if role == Role::Vector {
                n_per * 4
            } else {
                fmt_bytes(fmt, n_per)
            };
            let total = per_mat * n_mats as usize;
            let mut buf = AlignedBuf::zeroed(total).map_err(|_| {
                err!(
                    OutOfMemory,
                    "out of memory decoding tensor '{name}' ({:.1} MB)",
                    total as f64 / (1024.0 * 1024.0)
                )
            })?;
            let src_per_mat = (src_bytes / n_mats) as usize;
            let block_bytes = (src_bytes / (n / be)) as usize;
            for e in 0..n_mats as usize {
                let job = Decode {
                    ggml_type: gt.ggml_type,
                    src: &src[e * src_per_mat..(e + 1) * src_per_mat],
                    block_bytes,
                    block_elems: be as usize,
                    n: n_per,
                    to_f32: role == Role::Vector,
                    fmt,
                };
                if !decode(&job, &mut buf.as_mut_slice()[e * per_mat..(e + 1) * per_mat]) {
                    bail!(UnsupportedModel, "cannot decode tensor '{name}'");
                }
            }
            let p = buf.as_ptr();
            self.decoded_bytes += total;
            self.buffers.push(buf);
            p
        };
        self.used[idx] = true;
        if !self.prog.advance(src_bytes) {
            bail!(Cancelled, "loading was cancelled");
        }
        Ok(Some(Tensor::new(dtype, &ne, data)))
    }
}

// ---------------------------------------------------------------------------
// Shape checks. The kernels trust these shapes, so a file that breaks them is
// refused at load rather than read out of bounds later.
// ---------------------------------------------------------------------------

fn check_vector(t: &Option<Tensor>, what: &str, li: Option<usize>, len: usize) -> Result<()> {
    let Some(t) = t else { return Ok(()) };
    if t.n_dims != 1 || t.ne[0] != len {
        match li {
            Some(li) => bail!(BadFormat, "blk.{li}.{what} has {} values, expected {len}", t.ne[0]),
            None => bail!(BadFormat, "{what} has {} values, expected {len}", t.ne[0]),
        }
    }
    Ok(())
}

/// `rows` None: any number of rows.
fn check_matrix(t: &Option<Tensor>, what: &str, li: Option<usize>, cols: usize, rows: Option<usize>) -> Result<()> {
    let Some(t) = t else { return Ok(()) };
    if t.n_dims != 2 || t.ne[0] != cols || rows.is_some_and(|r| t.ne[1] != r) {
        let expect = rows.map_or("-1".to_string(), |r| r.to_string());
        match li {
            Some(li) => {
                bail!(
                    BadFormat,
                    "blk.{li}.{what} is [{}, {}], expected [{cols}, {expect}]",
                    t.ne[0],
                    t.ne[1]
                )
            }
            None => bail!(
                BadFormat,
                "{what} is [{}, {}], expected [{cols}, {expect}]",
                t.ne[0],
                t.ne[1]
            ),
        }
    }
    Ok(())
}

fn require(t: &Option<Tensor>, li: usize, what: &str) -> Result<()> {
    if t.is_none() {
        bail!(UnsupportedModel, "the model has no blk.{li}.{what}.weight");
    }
    Ok(())
}

fn check_experts(t: &Option<Tensor>, what: &str, li: usize, cols: usize, rows: usize, n_expert: usize) -> Result<()> {
    let Some(t) = t else { return Ok(()) };
    if t.n_dims != 3 || t.ne[0] != cols || t.ne[1] != rows || t.ne[2] != n_expert {
        bail!(
            BadFormat,
            "blk.{li}.{what} is [{}, {}, {}], expected [{cols}, {rows}, {n_expert}]",
            t.ne[0],
            t.ne[1],
            t.ne[2]
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Hyperparameters.
// ---------------------------------------------------------------------------

fn read_hparams(f: &GgufFile, h: &mut Hparams) -> Result<()> {
    let arch = h.arch.clone();
    let akey = |suffix: &str| format!("{arch}.{suffix}");

    let n_ctx = kv_int(f, &akey("context_length"));
    let n_embd = kv_int(f, &akey("embedding_length"));
    let n_layer = kv_int(f, &akey("block_count"));
    let n_head = kv_int(f, &akey("attention.head_count"));
    let (Some(n_ctx), Some(n_embd @ 1..), Some(n_layer @ 1..), Some(n_head @ 1..)) = (n_ctx, n_embd, n_layer, n_head)
    else {
        bail!(
            BadFormat,
            "the file lacks the basic {arch} dimensions (context length, embedding length, block count, head count)"
        );
    };
    h.n_ctx = if n_ctx > 0 { n_ctx as usize } else { 4096 };
    h.n_embd = n_embd as usize;
    h.n_layer = n_layer as usize;
    h.n_head = n_head as usize;
    // Every block has tensors of its own. Checked now, before the layer table
    // is allocated, so a corrupt count cannot ask for terabytes.
    if h.n_layer > f.tensors.len() {
        bail!(
            BadFormat,
            "the file claims {} blocks but holds only {} tensors",
            h.n_layer,
            f.tensors.len()
        );
    }

    // A Qwen3.5 file may carry its multi-token-prediction block after the
    // trunk, counted in block_count. nncpu does not run it.
    if h.arch_id == Arch::Qwen35
        && let Some(n_nextn) = kv_pos(f, &akey("nextn_predict_layers"))
        && n_nextn < h.n_layer
    {
        h.n_layer -= n_nextn;
    }
    h.n_head_kv = kv_pos(f, &akey("attention.head_count_kv")).unwrap_or(h.n_head);

    // llama doesn't publish key_length; derive from rope.dimension_count or n_embd/n_head.
    h.head_dim = match kv_pos(f, &akey("attention.key_length")) {
        Some(d) => d,
        None => kv_pos(f, &akey("rope.dimension_count")).unwrap_or(h.n_embd / h.n_head),
    };
    h.rope_freq_base = kv_float(f, &akey("rope.freq_base"))
        .filter(|&v| v > 0.0)
        .unwrap_or(10000.0);
    h.rms_eps = kv_float(f, &akey("attention.layer_norm_rms_epsilon"))
        .filter(|&v| v > 0.0)
        .unwrap_or(1e-6);

    h.head_dim_swa = kv_pos(f, &akey("attention.key_length_swa")).unwrap_or(h.head_dim);
    h.rope_dim = kv_pos(f, &akey("rope.dimension_count")).unwrap_or(h.head_dim); // gemma3 omits this
    h.rope_dim_swa = kv_pos(f, &akey("rope.dimension_count_swa")).unwrap_or(h.rope_dim);
    h.rope_freq_base_swa = match kv_float(f, &akey("rope.freq_base_swa")).filter(|&v| v > 0.0) {
        Some(v) => v,
        // gemma3 uses a hardcoded local-rope theta of 10000 for sliding layers
        // (HF: rope_local_base_freq); it is not stored in the GGUF.
        None if h.arch_id == Arch::Gemma3 => 10000.0,
        None => h.rope_freq_base,
    };
    h.sliding_window = kv_int(f, &akey("attention.sliding_window")).unwrap_or(0).max(0) as usize;
    let shared_kv = kv_int(f, &akey("attention.shared_kv_layers")).unwrap_or(0);
    h.ple_dim = kv_int(f, &akey("embedding_length_per_layer_input")).unwrap_or(0).max(0) as usize;
    h.n_expert = kv_int(f, &akey("expert_count")).unwrap_or(0).max(0) as usize;
    h.n_expert_used = kv_int(f, &akey("expert_used_count")).unwrap_or(0).max(0) as usize;
    if let Some(v) = kv_float(f, &akey("final_logit_softcapping")) {
        h.final_logit_softcap = v;
    }
    h.file_type = kv_int(f, "general.file_type").unwrap_or(0);

    if !h.causal {
        if kv_bool(f, &akey("attention.causal")) == Some(true) {
            bail!(UnsupportedModel, "this {arch} file asks for causal attention");
        }
        // llama.cpp's pooling enum: 0 none, 1 mean, 2 cls, 3 last, 4 rank.
        let pooling = kv_int(f, &akey("pooling_type")).unwrap_or(1);
        h.pooling = match pooling {
            1 => Pooling::Mean,
            2 => Pooling::Cls,
            3 => Pooling::Last,
            _ => bail!(
                UnsupportedModel,
                "the model pools its output with method {pooling}; nncpu does mean, first-token and last-token pooling"
            ),
        };
    }

    h.shared_kv_layers = if shared_kv < 0 || shared_kv as usize > h.n_layer.saturating_sub(2) {
        0
    } else {
        shared_kv as usize
    };
    h.n_layer_kv = h.n_layer - h.shared_kv_layers;
    if h.n_embd_target > 0 {
        h.n_layer_kv = 0; // a drafter reads its target's cache
    }
    // A bidirectional model's layers share one K/V buffer, each rewriting it
    // before it attends (kv_init), so no layer can read another's.
    if !h.causal && h.shared_kv_layers > 0 {
        bail!(
            UnsupportedModel,
            "this {arch} model shares K/V between layers, which nncpu does not support for bidirectional models"
        );
    }

    h.attention_scale = match kv_float(f, &akey("attention.scale")).filter(|&v| v > 0.0) {
        Some(v) => v,
        // Gemma 4 folds the scale into its q_norm (1.0). Gemma 3 27B scales by
        // 1/sqrt(n_embd / n_head) rather than 1/sqrt(head_dim), per
        // gemma_pytorch's config; everything else uses 1/sqrt(head_dim).
        None if h.arch_id == Arch::Gemma4 => 1.0,
        None if h.arch_id == Arch::Gemma3 && h.n_layer == 62 => 1.0 / ((h.n_embd / h.n_head) as f32).sqrt(),
        None => 1.0 / (h.head_dim as f32).sqrt(),
    };

    // RoPE linear scaling: theta_i is scaled by 1 / factor (effective_pos =
    // pos / factor). Both gemma3-4B and recent llama finetunes use this.
    let scaling_type = f.get_str(&akey("rope.scaling.type")).unwrap_or("");
    let factor = kv_float(f, &akey("rope.scaling.factor")).unwrap_or(0.0);
    if scaling_type == "linear" && factor > 0.0 {
        h.rope_freq_scale = 1.0 / factor;
    }

    // Llama's GGUF converter permutes Q/K for adjacent-pair RoPE; Gemma and
    // Qwen keep the half-split (NeoX) layout. Qwen3.5's interleaved
    // multimodal RoPE is NeoX RoPE over its first rope_dim lanes when, as for
    // text, every position component is the token's position.
    h.rope_mode = if h.arch_id == Arch::Llama {
        RopeMode::Norm
    } else {
        RopeMode::Neox
    };
    let gemma = matches!(h.arch_id, Arch::Gemma3 | Arch::Gemma4);
    h.embd_scale = if gemma { (h.n_embd as f32).sqrt() } else { 1.0 };

    if h.arch_id == Arch::Qwen35 {
        let d_conv = kv_int(f, &akey("ssm.conv_kernel"));
        let d_k = kv_int(f, &akey("ssm.state_size"));
        let n_k = kv_int(f, &akey("ssm.group_count"));
        let n_v = kv_int(f, &akey("ssm.time_step_rank"));
        let d_inner = kv_int(f, &akey("ssm.inner_size"));
        let ok = match (d_conv, d_k, n_k, n_v, d_inner) {
            (Some(d_conv @ 1..=16), Some(d_k @ 1..=1024), Some(n_k @ 1..), Some(n_v @ 1..), Some(d_inner @ 1..))
                if n_v % n_k == 0 && d_inner % n_v == 0 =>
            {
                h.ssm_d_conv = d_conv as usize;
                h.ssm_d_k = d_k as usize;
                h.ssm_n_k = n_k as usize;
                h.ssm_n_v = n_v as usize;
                h.ssm_d_v = (d_inner / n_v) as usize;
                true
            }
            _ => false,
        };
        if !ok {
            bail!(
                BadFormat,
                "the file lacks or garbles the Gated DeltaNet dimensions ({arch}.ssm.*)"
            );
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Per-layer resolution.
// ---------------------------------------------------------------------------

/// A Gated DeltaNet layer's tensors, checked against the dimensions in
/// hparams. Its convolution taps are transposed here, once, into the
/// time-major order conv1d_silu reads.
fn resolve_delta_net(h: &Hparams, l: &mut Layer, li: usize) -> Result<usize> {
    let parts = [
        ("attn_qkv", &l.ssm_qkv),
        ("attn_gate", &l.ssm_gate),
        ("ssm_alpha", &l.ssm_alpha),
        ("ssm_beta", &l.ssm_beta),
        ("ssm_conv1d", &l.ssm_conv1d),
        ("ssm_dt", &l.ssm_dt),
        ("ssm_a", &l.ssm_a),
        ("ssm_norm", &l.ssm_norm),
        ("ssm_out", &l.ssm_out),
    ];
    for (name, t) in parts {
        if t.is_none() {
            bail!(
                UnsupportedModel,
                "the model has no blk.{li}.{name} tensor for its Gated DeltaNet layer"
            );
        }
    }
    let n_embd = h.n_embd;
    let v_dim = h.ssm_n_v * h.ssm_d_v;
    let conv_dim = 2 * h.ssm_n_k * h.ssm_d_k + v_dim;
    let li_ = Some(li);
    check_matrix(&l.ssm_qkv, "attn_qkv", li_, n_embd, Some(conv_dim))?;
    check_matrix(&l.ssm_gate, "attn_gate", li_, n_embd, Some(v_dim))?;
    check_matrix(&l.ssm_alpha, "ssm_alpha", li_, n_embd, Some(h.ssm_n_v))?;
    check_matrix(&l.ssm_beta, "ssm_beta", li_, n_embd, Some(h.ssm_n_v))?;
    check_matrix(&l.ssm_conv1d, "ssm_conv1d", li_, h.ssm_d_conv, Some(conv_dim))?;
    check_vector(&l.ssm_dt, "ssm_dt", li_, h.ssm_n_v)?;
    check_vector(&l.ssm_a, "ssm_a", li_, h.ssm_n_v)?;
    check_vector(&l.ssm_norm, "ssm_norm", li_, h.ssm_d_v)?;
    check_matrix(&l.ssm_out, "ssm_out", li_, v_dim, Some(n_embd))?;

    let w = l.ssm_conv1d.as_ref().unwrap().f32s();
    let d_conv = h.ssm_d_conv;
    let mut taps = Vec::new();
    taps.try_reserve_exact(d_conv * conv_dim)
        .map_err(|_| err!(OutOfMemory, "out of memory loading blk.{li}.ssm_conv1d"))?;
    taps.resize(d_conv * conv_dim, 0.0f32);
    for c in 0..conv_dim {
        for j in 0..d_conv {
            taps[j * conv_dim + c] = w[c * d_conv + j];
        }
    }
    let bytes = taps.len() * 4;
    l.ssm_conv_taps = taps;
    l.kv_layer = None;
    Ok(bytes)
}

/// An attention layer's heads, and the K/V cache it reads. `before` holds the
/// layers below this one.
fn resolve_attention(
    f: Option<&GgufFile>,
    h: &Hparams,
    rope_freqs: &Option<Tensor>,
    before: &[Layer],
    l: &mut Layer,
    li: usize,
) -> Result<()> {
    let n_embd = h.n_embd;
    require(&l.attn_q, li, "attn_q")?;
    require(&l.attn_output, li, "attn_output")?;
    let owns_kv = li < h.n_layer_kv;
    if owns_kv {
        require(&l.attn_k, li, "attn_k")?;
    }

    l.head_dim = match &l.attn_q_norm {
        Some(t) => t.ne[0],
        None if l.sliding_window => h.head_dim_swa,
        None => h.head_dim,
    };
    if l.head_dim == 0 || !l.head_dim.is_multiple_of(2) || l.head_dim > 512 {
        bail!(
            UnsupportedModel,
            "blk.{li} has head size {}; nncpu supports even sizes up to 512",
            l.head_dim
        );
    }
    let li_ = Some(li);
    check_matrix(&l.attn_q, "attn_q", li_, n_embd, None)?;
    // Qwen3.5's projection gives each head its query, then a gate for the
    // head's output.
    l.attn_gated = h.arch_id == Arch::Qwen35;
    let per_head = if l.attn_gated { 2 * l.head_dim } else { l.head_dim };
    let q_rows = l.attn_q.as_ref().unwrap().ne[1];
    if !q_rows.is_multiple_of(per_head) {
        bail!(
            BadFormat,
            "blk.{li}.attn_q has {q_rows} outputs, not whole heads of {per_head}"
        );
    }
    l.n_head = q_rows / per_head;
    let q_dim = l.n_head * l.head_dim;

    let kv_dim;
    if owns_kv {
        check_matrix(&l.attn_k, "attn_k", li_, n_embd, None)?;
        kv_dim = l.attn_k.as_ref().unwrap().ne[1];
        check_matrix(&l.attn_v, "attn_v", li_, n_embd, Some(kv_dim))?;
        l.kv_layer = Some(li);
    } else if h.n_embd_target > 0 {
        // An MTP drafter reads its target's cache; draft_init checks that
        // these shapes match it.
        let key = format!("{}.attention.head_count_kv", h.arch);
        let n_kv = f
            .and_then(|f| kv_int_at(f, &key, li))
            .map_or(h.n_head_kv, |v| v.max(0) as usize);
        l.kv_layer = None;
        kv_dim = n_kv * l.head_dim;
    } else {
        // Shared KV (Gemma 4): the trailing layers read the cache of the last
        // cache-owning layer of the same kind, global or sliding.
        let back = if l.sliding_window { 2 } else { 1 };
        let kv_layer = h.n_layer_kv.checked_sub(back);
        let Some(kv_layer) = kv_layer.filter(|&k| before[k].sliding_window == l.sliding_window) else {
            bail!(
                UnsupportedModel,
                "blk.{li} shares a KV cache in a layer pattern nncpu does not recognise"
            );
        };
        let s = &before[kv_layer];
        if s.head_dim != l.head_dim {
            bail!(UnsupportedModel, "blk.{li} shares a KV cache of a different head size");
        }
        l.kv_layer = Some(kv_layer);
        kv_dim = s.n_head_kv * s.head_dim;
    }
    if kv_dim == 0 || !kv_dim.is_multiple_of(l.head_dim) {
        bail!(
            BadFormat,
            "blk.{li} has {kv_dim} K/V outputs, not whole heads of {}",
            l.head_dim
        );
    }
    l.n_head_kv = kv_dim / l.head_dim;
    if !l.n_head.is_multiple_of(l.n_head_kv) {
        bail!(
            BadFormat,
            "blk.{li} has {} query heads for {} KV heads",
            l.n_head,
            l.n_head_kv
        );
    }

    check_matrix(&l.attn_output, "attn_output", li_, q_dim, Some(n_embd))?;
    check_vector(&l.attn_q_norm, "attn_q_norm", li_, l.head_dim)?;
    check_vector(&l.attn_k_norm, "attn_k_norm", li_, l.head_dim)?;
    check_vector(&l.attn_q_b, "attn_q.bias", li_, q_rows)?;
    check_vector(&l.attn_k_b, "attn_k.bias", li_, kv_dim)?;
    check_vector(&l.attn_v_b, "attn_v.bias", li_, kv_dim)?;

    let n_rot = if l.sliding_window { h.rope_dim_swa } else { h.rope_dim };
    l.n_rot = n_rot.min(l.head_dim) & !1;

    // RoPE frequency factors: Llama 3's long-context scaling on every layer,
    // Gemma 4's proportional RoPE on its global layers only.
    let wants_freqs = h.arch_id == Arch::Llama || (h.arch_id == Arch::Gemma4 && !l.sliding_window);
    if wants_freqs
        && let Some(rf) = rope_freqs
        && rf.ne[0] == l.n_rot / 2
    {
        l.rope_freqs = Some(rf.clone());
    }
    Ok(())
}

/// Per-layer shapes, from the tensors themselves (some GGUFs disagree with
/// their own key_length), checked against everything the forward pass will
/// assume. Returns the bytes of data decoded here.
pub(super) fn resolve_layers(
    f: Option<&GgufFile>,
    h: &Hparams,
    rope_freqs: &Option<Tensor>,
    layers: &mut [Layer],
) -> Result<usize> {
    let n_embd = h.n_embd;
    let mut decoded = 0;
    for li in 0..layers.len() {
        let (before, rest) = layers.split_at_mut(li);
        let l = &mut rest[0];
        require(&l.attn_norm, li, "attn_norm")?;
        require(&l.ffn_norm, li, "ffn_norm")?;
        require(&l.ffn_gate, li, "ffn_gate")?;
        require(&l.ffn_up, li, "ffn_up")?;
        require(&l.ffn_down, li, "ffn_down")?;
        if l.rs_index.is_some() {
            decoded += resolve_delta_net(h, l, li)?;
        } else {
            resolve_attention(f, h, rope_freqs, before, l, li)?;
        }

        let li_ = Some(li);
        check_matrix(&l.ffn_gate, "ffn_gate", li_, n_embd, None)?;
        l.n_ff = l.ffn_gate.as_ref().unwrap().ne[1];
        check_matrix(&l.ffn_up, "ffn_up", li_, n_embd, Some(l.n_ff))?;
        check_matrix(&l.ffn_down, "ffn_down", li_, l.n_ff, Some(n_embd))?;
        check_vector(&l.attn_norm, "attn_norm", li_, n_embd)?;
        check_vector(&l.ffn_norm, "ffn_norm", li_, n_embd)?;
        check_vector(&l.post_attention_norm, "post_attention_norm", li_, n_embd)?;
        check_vector(&l.post_ffw_norm, "post_ffw_norm", li_, n_embd)?;
        check_vector(&l.post_norm, "post_norm", li_, n_embd)?;
        if let Some(s) = &l.layer_output_scale
            && s.ne[0] < 1
        {
            bail!(BadFormat, "blk.{li}.layer_output_scale is empty");
        }
        if l.inp_gate.is_some() || l.proj.is_some() {
            if l.inp_gate.is_none() || l.proj.is_none() || h.ple_dim == 0 {
                bail!(BadFormat, "blk.{li} has half a per-layer-embedding block");
            }
            check_matrix(&l.inp_gate, "inp_gate", li_, n_embd, Some(h.ple_dim))?;
            check_matrix(&l.proj, "proj", li_, h.ple_dim, Some(n_embd))?;
        }

        if l.ffn_gate_inp.is_some()
            || l.ffn_gate_up_exps.is_some()
            || l.ffn_gate_exps.is_some()
            || l.ffn_up_exps.is_some()
            || l.ffn_down_exps.is_some()
        {
            if l.ffn_gate_inp.is_none()
                || l.ffn_gate_inp_scale.is_none()
                || l.ffn_down_exps.is_none()
                || !(l.ffn_gate_up_exps.is_some() || (l.ffn_gate_exps.is_some() && l.ffn_up_exps.is_some()))
                || l.pre_ffw_norm_2.is_none()
                || l.post_ffw_norm_1.is_none()
                || l.post_ffw_norm_2.is_none()
            {
                bail!(
                    UnsupportedModel,
                    "blk.{li} has a mixture-of-experts layout nncpu does not recognise"
                );
            }
            let n_expert = h.n_expert;
            if n_expert == 0 || n_expert > MAX_EXPERTS || h.n_expert_used == 0 || h.n_expert_used > n_expert {
                bail!(
                    UnsupportedModel,
                    "the model routes each token to {} of {n_expert} experts; nncpu supports up to {MAX_EXPERTS}",
                    h.n_expert_used
                );
            }
            let n_ff_exp = l.ffn_down_exps.as_ref().unwrap().ne[0];
            if n_ff_exp == 0 {
                bail!(BadFormat, "blk.{li} has empty experts");
            }
            check_matrix(&l.ffn_gate_inp, "ffn_gate_inp", li_, n_embd, Some(n_expert))?;
            check_vector(&l.ffn_gate_inp_scale, "ffn_gate_inp.scale", li_, n_embd)?;
            check_experts(
                &l.ffn_gate_up_exps,
                "ffn_gate_up_exps",
                li,
                n_embd,
                2 * n_ff_exp,
                n_expert,
            )?;
            check_experts(&l.ffn_gate_exps, "ffn_gate_exps", li, n_embd, n_ff_exp, n_expert)?;
            check_experts(&l.ffn_up_exps, "ffn_up_exps", li, n_embd, n_ff_exp, n_expert)?;
            check_experts(&l.ffn_down_exps, "ffn_down_exps", li, n_ff_exp, n_embd, n_expert)?;
            check_vector(&l.ffn_down_exps_scale, "ffn_down_exps.scale", li_, n_expert)?;
            check_vector(&l.pre_ffw_norm_2, "pre_ffw_norm_2", li_, n_embd)?;
            check_vector(&l.post_ffw_norm_1, "post_ffw_norm_1", li_, n_embd)?;
            check_vector(&l.post_ffw_norm_2, "post_ffw_norm_2", li_, n_embd)?;
            l.n_ff_exp = n_ff_exp;
        }
    }
    Ok(decoded)
}

// ---------------------------------------------------------------------------
// Loading.
// ---------------------------------------------------------------------------

/// Load a model. `progress` gets the fraction of the file processed and may
/// return false to cancel.
pub(crate) fn load(path: &Path, fmt: WeightFormat, progress: Option<ProgressFn>) -> Result<ModelData> {
    let t0 = Instant::now();
    let path_str = path.to_string_lossy().into_owned();
    let f = GgufFile::open(path)?;

    // --- architecture --------------------------------------------------
    let mut h = Hparams::default();
    let Some(arch) = f.get_str("general.architecture") else {
        bail!(
            UnsupportedModel,
            "'{path_str}' does not say what model it holds (no general.architecture)"
        );
    };
    h.arch = arch.to_string();
    h.arch_id = match arch {
        "gemma3" => Arch::Gemma3,
        "gemma4" => Arch::Gemma4,
        "llama" => Arch::Llama,
        "qwen2" => Arch::Qwen2,
        "qwen3" => Arch::Qwen3,
        "qwen35" => Arch::Qwen35,
        "gemma-embedding" | "gemma-embedding2" => {
            // EmbeddingGemma: Gemma 3 (or, for version 2, Gemma 4) layers
            // that attend both ways, pooled into one vector.
            h.causal = false;
            h.pooling = Pooling::Mean;
            if arch == "gemma-embedding" {
                Arch::Gemma3
            } else {
                Arch::Gemma4
            }
        }
        "gemma4-assistant" => {
            // Gemma 4's MTP drafter: query-only Gemma 4 layers over the target
            // model's K/V cache (draft_step).
            h.n_embd_target = match kv_int(&f, "gemma4-assistant.embedding_length_out") {
                Some(w) if w > 0 => w as usize,
                _ => bail!(BadFormat, "the drafter does not say how wide its target model is"),
            };
            Arch::Gemma4
        }
        "clip" => bail!(
            UnsupportedModel,
            "'{path_str}' is a vision projector (mmproj), not a language model"
        ),
        "qwen4exp" => bail!(
            UnsupportedModel,
            "'{path_str}' is Qwen3.8-Flash-Next (qwen4exp), whose sparse attention, hyper-connections, n-gram \
             embeddings and 512 experts nncpu does not implement; nncpu runs Qwen's dense models (qwen2, qwen3 and \
             qwen35, which includes Qwen3.8-27B)"
        ),
        "qwen2moe" | "qwen3moe" | "qwen35moe" | "qwen3next" => bail!(
            UnsupportedModel,
            "'{path_str}' is a mixture-of-experts Qwen model ({arch}); nncpu runs Qwen's dense models (qwen2, qwen3 \
             and qwen35) and Gemma 4's experts only"
        ),
        _ => bail!(
            UnsupportedModel,
            "'{path_str}' is a '{arch}' model; nncpu runs gemma3, gemma4, llama, qwen2, qwen3, qwen35, \
             gemma-embedding and gemma-embedding2"
        ),
    };
    let qwen = matches!(h.arch_id, Arch::Qwen2 | Arch::Qwen3 | Arch::Qwen35);

    // Weights rotated by a Hadamard transform need the rotation undone at run
    // time, which nncpu does not do.
    if f.get("prism.hadamard.version").is_some() {
        bail!(
            UnsupportedModel,
            "'{path_str}' holds Hadamard-rotated weights, which nncpu cannot run"
        );
    }
    // Qwen3-Embedding is a qwen3 model that pools its output into a vector.
    if qwen && f.get(&format!("{}.pooling_type", h.arch)).is_some() {
        bail!(
            UnsupportedModel,
            "'{path_str}' is a Qwen text-embedding model, which nncpu does not run"
        );
    }
    // Gemma 4's mixture of experts (a dense MLP plus routed experts) is the
    // only one nncpu runs; other architectures route differently.
    if h.arch_id != Arch::Gemma4
        && (kv_int(&f, &format!("{}.expert_count", h.arch)).is_some_and(|n| n > 0)
            || f.find_tensor("blk.0.ffn_gate_inp.weight").is_some())
    {
        bail!(
            UnsupportedModel,
            "'{path_str}' is a mixture-of-experts {} model; nncpu runs Gemma 4's experts only",
            h.arch
        );
    }

    let name = match f.get_str("general.name") {
        Some(n) if !n.is_empty() => n.to_string(),
        _ => path
            .file_name()
            .map_or_else(|| path_str.clone(), |n| n.to_string_lossy().into_owned()),
    };

    read_hparams(&f, &mut h)?;
    // llama.cpp's mtp-*.gguf files for Qwen3.5 hold only the trailing
    // multi-token-prediction block, with the embeddings it shares.
    if h.arch_id == Arch::Qwen35
        && f.find_tensor("blk.0.attn_norm.weight").is_none()
        && f.find_tensor(&format!("blk.{}.nextn.eh_proj.weight", h.n_layer))
            .is_some()
    {
        bail!(
            UnsupportedModel,
            "'{path_str}' holds only a Qwen multi-token-prediction drafter, which nncpu does not run"
        );
    }
    let vocab = Vocab::load(&f)?;
    h.n_vocab = vocab.n_vocab();

    // --- tensors -------------------------------------------------------
    let mut prog = Progress {
        f: progress,
        total: 0,
        done: 0,
        reported: -1.0,
        cancelled: false,
    };
    let mut n_params = 0u64;
    for t in &f.tensors {
        // Saturating: the descriptors are not yet checked against the data.
        prog.total = prog.total.saturating_add(t.nbytes());
        n_params = n_params.saturating_add(t.nelements());
    }
    f.prefetch();
    if !prog.advance(0) {
        bail!(Cancelled, "loading was cancelled");
    }

    let mut ld = Loader {
        f: &f,
        fmt,
        prog,
        used: vec![false; f.tensors.len()],
        buffers: Vec::new(),
        decoded_bytes: 0,
    };
    use Role::{Experts, Matrix, Vector};

    // A generative model's output.weight is its lm_head. An embedding model
    // has no lm_head: EmbeddingGemma 2's output.weight projects the pooled
    // state, and EmbeddingGemma's projections are dense_2 and dense_3.
    let embedding = !h.causal;
    let token_embd = ld.load("token_embd.weight", Matrix)?;
    let output_norm = ld.load("output_norm.weight", Vector)?;
    let output = if embedding {
        None
    } else {
        ld.load("output.weight", Matrix)?
    };
    let rope_freqs = ld.load("rope_freqs.weight", Vector)?;
    let per_layer_token_embd = ld.load("per_layer_token_embd.weight", Matrix)?;
    let per_layer_model_proj = ld.load("per_layer_model_proj.weight", Matrix)?;
    let per_layer_proj_norm = ld.load("per_layer_proj_norm.weight", Vector)?;
    if token_embd.is_none() {
        bail!(UnsupportedModel, "the model has no token_embd.weight");
    }
    check_matrix(&token_embd, "token_embd.weight", None, h.n_embd, Some(h.n_vocab))?;
    check_matrix(&output, "output.weight", None, h.n_embd, Some(h.n_vocab))?;
    check_vector(&output_norm, "output_norm.weight", None, h.n_embd)?;

    let (mut nextn_pre, mut nextn_post) = (None, None);
    if h.n_embd_target > 0 {
        nextn_pre = ld.load("nextn.pre_projection.weight", Matrix)?;
        nextn_post = ld.load("nextn.post_projection.weight", Matrix)?;
        if nextn_pre.is_none() || nextn_post.is_none() || output_norm.is_none() {
            bail!(UnsupportedModel, "the drafter lacks its input or output projection");
        }
        check_matrix(
            &nextn_pre,
            "nextn.pre_projection.weight",
            None,
            2 * h.n_embd_target,
            Some(h.n_embd),
        )?;
        check_matrix(
            &nextn_post,
            "nextn.post_projection.weight",
            None,
            h.n_embd,
            Some(h.n_embd_target),
        )?;
    }

    let mut embd_proj = Vec::new();
    if embedding {
        let v2 = h.arch_id == Arch::Gemma4;
        let names: &[&str] = if v2 {
            &["output.weight"]
        } else {
            &["dense_2.weight", "dense_3.weight"]
        };
        let mut width = h.n_embd;
        for &name in names {
            let t = ld.load(name, Matrix)?;
            if t.is_none() {
                continue; // EmbeddingGemma converted without its dense layers
            }
            check_matrix(&t, name, None, width, None)?;
            let t = t.unwrap();
            width = t.ne[1];
            embd_proj.push(t);
        }
        if v2 && embd_proj.is_empty() {
            bail!(UnsupportedModel, "the model has no output.weight projection");
        }
        if let Some(declared) = kv_int(&f, &format!("{}.embedding_length_out", h.arch))
            && declared as i64 != width as i64
        {
            bail!(
                BadFormat,
                "the model says its embeddings have {declared} values, but its projections make {width}"
            );
        }
        h.n_embd_out = width;
    }

    // The per-token table is optional: EmbeddingGemma 2 derives its
    // per-layer inputs from the projection alone.
    let have_ple = per_layer_token_embd.is_some() || per_layer_model_proj.is_some() || per_layer_proj_norm.is_some();
    if have_ple {
        let ple_total = h.ple_dim * h.n_layer;
        if h.ple_dim == 0 || per_layer_model_proj.is_none() || per_layer_proj_norm.is_none() {
            bail!(BadFormat, "the per-layer embedding tables are incomplete");
        }
        check_matrix(
            &per_layer_token_embd,
            "per_layer_token_embd.weight",
            None,
            ple_total,
            Some(h.n_vocab),
        )?;
        check_matrix(
            &per_layer_model_proj,
            "per_layer_model_proj.weight",
            None,
            h.n_embd,
            Some(ple_total),
        )?;
        check_vector(&per_layer_proj_norm, "per_layer_proj_norm.weight", None, h.ple_dim)?;
    } else {
        h.ple_dim = 0;
    }

    let mut layers: Vec<Layer> = (0..h.n_layer).map(|_| Layer::default()).collect();
    // Qwen3.5: which layers are Gated DeltaNet layers. Newer files list them;
    // older ones give the period of the attention layers instead.
    if h.arch_id == Arch::Qwen35 {
        let interval = kv_int(&f, &format!("{}.full_attention_interval", h.arch)).unwrap_or(4);
        let key = format!("{}.attention.recurrent_layers", h.arch);
        for (li, l) in layers.iter_mut().enumerate() {
            let mut recurrent = interval > 0 && (li as i64 + 1) % interval as i64 != 0;
            if let Some(r) = kv_flag_at(&f, &key, li) {
                recurrent = r;
            }
            if recurrent {
                l.rs_index = Some(h.n_layer_rs);
                h.n_layer_rs += 1;
            }
        }
    }
    let swp_key = format!("{}.attention.sliding_window_pattern", h.arch);
    let swp = f.get(&swp_key);
    let swa_period = match swp {
        Some(GgufValue::Array(..)) | None => 6,
        Some(_) => kv_int(&f, &swp_key).unwrap_or(6),
    };
    for (li, l) in layers.iter_mut().enumerate() {
        match swp {
            Some(GgufValue::Array(_, a)) if a.len() == h.n_layer => l.sliding_window = a[li].to_i64_or_zero() != 0,
            _ if (h.arch_id == Arch::Gemma3 || embedding) && h.sliding_window > 0 => {
                // Every sixth layer is global; the rest slide. Matches Hugging
                // Face's `is_sliding = bool((i + 1) % sliding_window_pattern)`.
                l.sliding_window = swa_period <= 0 || (li as i64 + 1) % swa_period as i64 != 0;
            }
            _ => {}
        }
        if h.sliding_window == 0 {
            l.sliding_window = false;
        }

        let p = format!("blk.{li}.");
        let t = |s: &str| format!("{p}{s}");
        l.attn_norm = ld.load(&t("attn_norm.weight"), Vector)?;
        l.ffn_norm = ld.load(&t("ffn_norm.weight"), Vector)?;
        l.post_attention_norm = ld.load(&t("post_attention_norm.weight"), Vector)?;
        l.post_ffw_norm = ld.load(&t("post_ffw_norm.weight"), Vector)?;
        l.post_norm = ld.load(&t("post_norm.weight"), Vector)?;
        l.attn_q_norm = ld.load(&t("attn_q_norm.weight"), Vector)?;
        l.attn_k_norm = ld.load(&t("attn_k_norm.weight"), Vector)?;
        l.layer_output_scale = ld.load(&t("layer_output_scale.weight"), Vector)?;
        l.attn_q = ld.load(&t("attn_q.weight"), Matrix)?;
        l.attn_output = ld.load(&t("attn_output.weight"), Matrix)?;
        l.ffn_gate = ld.load(&t("ffn_gate.weight"), Matrix)?;
        l.ffn_up = ld.load(&t("ffn_up.weight"), Matrix)?;
        l.ffn_down = ld.load(&t("ffn_down.weight"), Matrix)?;
        l.inp_gate = ld.load(&t("inp_gate.weight"), Matrix)?;
        l.proj = ld.load(&t("proj.weight"), Matrix)?;
        l.ffn_gate_inp = ld.load(&t("ffn_gate_inp.weight"), Vector)?;
        l.ffn_gate_inp_scale = ld.load(&t("ffn_gate_inp.scale"), Vector)?;
        l.ffn_gate_up_exps = ld.load(&t("ffn_gate_up_exps.weight"), Experts)?;
        l.ffn_gate_exps = ld.load(&t("ffn_gate_exps.weight"), Experts)?;
        l.ffn_up_exps = ld.load(&t("ffn_up_exps.weight"), Experts)?;
        l.ffn_down_exps = ld.load(&t("ffn_down_exps.weight"), Experts)?;
        l.ffn_down_exps_scale = ld.load(&t("ffn_down_exps.scale"), Vector)?;
        l.pre_ffw_norm_2 = ld.load(&t("pre_ffw_norm_2.weight"), Vector)?;
        l.post_ffw_norm_1 = ld.load(&t("post_ffw_norm_1.weight"), Vector)?;
        l.post_ffw_norm_2 = ld.load(&t("post_ffw_norm_2.weight"), Vector)?;
        // Layers that reuse an earlier layer's KV never read their own K/V
        // projections, so those are not loaded at all.
        if li < h.n_layer_kv {
            l.attn_k = ld.load(&t("attn_k.weight"), Matrix)?;
            l.attn_v = ld.load(&t("attn_v.weight"), Matrix)?;
        }
        if l.rs_index.is_some() {
            // The decay and write-strength rows are few, and steer the state
            // for the rest of the sequence, so they stay F32 like a router.
            l.ssm_qkv = ld.load(&t("attn_qkv.weight"), Matrix)?;
            l.ssm_gate = ld.load(&t("attn_gate.weight"), Matrix)?;
            l.ssm_alpha = ld.load(&t("ssm_alpha.weight"), Vector)?;
            l.ssm_beta = ld.load(&t("ssm_beta.weight"), Vector)?;
            l.ssm_conv1d = ld.load(&t("ssm_conv1d.weight"), Vector)?;
            l.ssm_dt = ld.load(&t("ssm_dt.bias"), Vector)?;
            l.ssm_a = ld.load(&t("ssm_a"), Vector)?;
            l.ssm_norm = ld.load(&t("ssm_norm.weight"), Vector)?;
            l.ssm_out = ld.load(&t("ssm_out.weight"), Matrix)?;
        } else {
            l.attn_q_b = ld.load(&t("attn_q.bias"), Vector)?;
            l.attn_k_b = ld.load(&t("attn_k.bias"), Vector)?;
            l.attn_v_b = ld.load(&t("attn_v.bias"), Vector)?;
        }
        // Qwen3.5 calls its pre-MLP norm post_attention_norm; Gemma's norm of
        // that name scales the attention output instead.
        if h.arch_id == Arch::Qwen35 {
            l.ffn_norm = l.post_attention_norm.take();
        }
    }
    ld.decoded_bytes += resolve_layers(Some(&f), &h, &rope_freqs, &mut layers)?;
    if qwen {
        for (i, t) in f.tensors.iter().enumerate() {
            // Blocks past the last layer are Qwen3.5's multi-token predictor.
            let past_trunk = t
                .name
                .strip_prefix("blk.")
                .and_then(|r| r.split('.').next())
                .and_then(|n| n.parse::<usize>().ok())
                .is_some_and(|li| li >= h.n_layer);
            if ld.used[i] || past_trunk {
                continue;
            }
            bail!(
                UnsupportedModel,
                "'{path_str}' holds a tensor nncpu does not know how to use ({})",
                t.name
            );
        }
    }
    let Loader {
        mut prog,
        buffers,
        decoded_bytes,
        ..
    } = ld;
    if !prog.cancelled
        && prog.reported < 1.0
        && let Some(cb) = prog.f.as_mut()
    {
        cb(1.0);
    }

    log::info!(
        "loaded {name}: {}, {} layers, {}-token vocabulary, {:.1} MB of {} weights decoded in {:.1} s",
        h.arch,
        h.n_layer,
        h.n_vocab,
        decoded_bytes as f64 / (1024.0 * 1024.0),
        fmt.name(),
        t0.elapsed().as_secs_f64()
    );

    Ok(ModelData {
        path: path_str,
        name,
        gguf: Some(f),
        hparams: h,
        vocab,
        token_embd,
        per_layer_token_embd,
        per_layer_model_proj,
        per_layer_proj_norm,
        output_norm,
        output,
        rope_freqs,
        embd_proj,
        nextn_pre,
        nextn_post,
        layers,
        buffers,
        weights: fmt,
        decoded_bytes,
        n_params,
    })
}
