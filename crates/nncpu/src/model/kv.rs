//! The key/value cache, recurrent state and forward-pass scratch.

use super::{ATTN_BLOCK, ModelData};
use crate::error::{Result, bail, err};
use crate::kernels::{self, KvRows};
use crate::pool;

/// One layer's K or V storage, token-major: BF16 (the default) or F32.
pub(crate) enum KvVec {
    Bf16(Vec<u16>),
    F32(Vec<f32>),
}

impl Default for KvVec {
    fn default() -> Self {
        KvVec::F32(Vec::new())
    }
}

impl KvVec {
    /// `len` elements from element `start`.
    #[inline]
    pub fn rows(&self, start: usize, len: usize) -> KvRows<'_> {
        match self {
            KvVec::Bf16(v) => KvRows::Bf16(&v[start..start + len]),
            KvVec::F32(v) => KvRows::F32(&v[start..start + len]),
        }
    }
}

/// One layer's K/V storage: position `pos` is row `pos % rows`, kv_dim
/// elements. A full buffer has n_ctx rows. A causal sliding-window layer keeps
/// only a ring of round_up(window + n_batch, ATTN_BLOCK) rows, which holds
/// every key a query of a batch can see, plus a margin for going back (see
/// kv_rewind).
#[derive(Default)]
pub(crate) struct KvBuffer {
    pub k: KvVec,
    pub v: KvVec,
    pub rows: usize,
    /// A ring's window; 0 for a full buffer.
    pub window: usize,
}

/// Forward-pass scratch, sized once in kv_init to the maxima across all
/// layers and n_batch tokens. Nothing is allocated per token.
#[derive(Default)]
pub(crate) struct Scratch {
    /// [n_batch, n_embd] residual stream / final hidden.
    pub x: Vec<f32>,
    /// [n_batch, n_embd]
    pub normed: Vec<f32>,
    /// [n_batch, max_q_dim]
    pub q: Vec<f32>,
    /// [n_batch, max_kv_dim]
    pub k_cur: Vec<f32>,
    /// [n_batch, max_kv_dim]
    pub v_cur: Vec<f32>,
    /// [n_batch, max_q_dim]
    pub attn: Vec<f32>,
    /// [n_batch, max_ff]
    pub gate: Vec<f32>,
    /// [n_batch, max_ff]
    pub up: Vec<f32>,
    /// n_attn_slots x attn_slot: block scores, then two partials.
    pub scores: Vec<f32>,
    /// [n_head_max * n_blocks * attn_part(head_dim)]: decode's per-block partials.
    pub attn_part: Vec<f32>,
    /// [n_batch, ple_dim]
    pub ple_gate: Vec<f32>,
    /// [n_batch, n_embd]
    pub ple_contrib: Vec<f32>,
    /// [n_batch, ple_dim * n_layer]
    pub ple_inputs: Vec<f32>,
    /// [n_batch, ple_dim * n_layer]; empty without per_layer_token_embd.
    pub ple_proj: Vec<f32>,
    /// [n_batch, max_cols/32 + 1]: Q4 per-block sums of x.
    pub xsum: Vec<f32>,
    /// [n_batch, 2 * max_q_dim]: gated attention's [q | gate] per head, as projected.
    pub q_gate: Vec<f32>,
    /// [n_batch, max_q_dim]: ...and its gate half.
    pub attn_gate: Vec<f32>,

    // Gated DeltaNet scratch; empty for other models.
    /// [d_conv - 1 + n_batch, conv_dim]: convolution input, history then the batch.
    pub dn_conv: Vec<f32>,
    /// [n_batch, conv_dim]: convolution output, q | k | v per token.
    pub dn_qkv: Vec<f32>,
    /// [n_batch, n_v * d_v]: output gate.
    pub dn_z: Vec<f32>,
    /// [n_batch, 2 * n_v]: each head's decay, then its write strength.
    pub dn_gb: Vec<f32>,
    /// [n_batch, n_v * d_v]
    pub dn_out: Vec<f32>,

    // Mixture-of-experts scratch; empty for dense models.
    /// [n_batch, n_embd]: expert input (pre_ffw_norm_2).
    pub moe_in: Vec<f32>,
    /// [n_batch, n_embd]: weighted sum of expert outputs.
    pub moe_out: Vec<f32>,
    /// [n_batch, n_embd]: inputs gathered for one expert.
    pub moe_rows: Vec<f32>,
    /// [n_batch, n_embd]: one expert's outputs.
    pub moe_down: Vec<f32>,
    /// [n_batch, 2 * max_ff_exp]
    pub moe_gate_up: Vec<f32>,
    /// [n_embd]: router input of one token.
    pub moe_router: Vec<f32>,
    /// [n_expert]
    pub moe_probs: Vec<f32>,
    /// [n_batch, n_expert_used]
    pub moe_weight: Vec<f32>,
    /// [n_batch, n_expert_used]: chosen experts.
    pub moe_expert: Vec<i32>,
    /// [n_batch]: the (token, slot) pairs that chose one expert.
    pub moe_pairs: Vec<usize>,

    // MTP drafter scratch; empty for other models.
    /// [2 * n_embd_target]: [target embedding | hidden].
    pub draft_in: Vec<f32>,
    /// [n_vocab]
    pub draft_logits: Vec<f32>,
}

/// Per-layer key/value cache plus all forward-pass scratch. K/V live in
/// `bufs[kv_buf[li]]`. kv_dim can differ per layer (Gemma 4 global layers
/// are wider than its sliding ones). Layers that reuse an earlier layer's
/// cache own no storage, and a bidirectional (embedding) model keeps one
/// buffer for all layers: each layer rewrites the whole sequence before it
/// attends. An MTP drafter's cache owns none either: its layers read its
/// target's cache, which `kv_buf` indexes, and is passed to the forward pass.
#[derive(Default)]
pub(crate) struct KvCache {
    /// Max positions.
    pub n_ctx: usize,
    /// Next slot to fill.
    pub cur_pos: usize,
    pub kv_dim_per_layer: Vec<usize>,
    /// [n_layer] the buffer in `bufs` each layer uses.
    pub kv_buf: Vec<Option<usize>>,
    pub bufs: Vec<KvBuffer>,
    /// Element type of every buffer.
    pub kv_bf16: bool,
    /// The highest position written since position 0, plus one.
    pub kv_written: usize,
    /// How far below kv_written cur_pos may go back; usize::MAX without rings.
    pub max_rewind: usize,

    // Recurrent state of the Gated DeltaNet layers (Qwen3.5), in slot
    // Layer::rs_index of each: the last d_conv - 1 inputs of the layer's
    // causal convolution, and its delta-rule matrices, [d_v][d_k] per value
    // head. Unlike K/V it sums up the whole history, so it always describes
    // exactly cur_pos tokens: model_forward() at position 0 clears it, and
    // going back means state_restore().
    /// [n_layer_rs][d_conv - 1][conv_dim]
    pub conv_state: Vec<f32>,
    /// [n_layer_rs][n_v][d_v][d_k]
    pub ssm_state: Vec<f32>,
    pub conv_saved: Vec<f32>,
    pub ssm_saved: Vec<f32>,
    /// cur_pos when they were saved; None: nothing saved.
    pub saved_pos: Option<usize>,

    pub s: Scratch,

    /// Most tokens one model_forward call may take.
    pub n_batch: usize,
    /// Query heads in the widest layer.
    pub n_head_max: usize,
    /// Score slots: how many chunks of attention work run at once.
    pub n_attn_slots: usize,
    /// Floats per slot.
    pub attn_slot: usize,
    /// The most n_head * attn_part(head_dim) of any layer.
    pub attn_part_max: usize,
    /// The most blocks one query's keys span.
    pub n_attn_blocks: usize,
    /// K/V storage, and the recurrent state.
    pub bytes: usize,
}

fn zeros<T: Clone + Default>(n: usize, what: impl Fn() -> String) -> Result<Vec<T>> {
    let mut v = Vec::new();
    v.try_reserve_exact(n).map_err(|_| err!(OutOfMemory, "{}", what()))?;
    v.resize(n, T::default());
    Ok(v)
}

/// The largest context a cache takes, as in the C API's `int n_ctx`.
pub(crate) const MAX_CTX: usize = i32::MAX as usize;

/// A cache for `n_ctx` positions, taking up to `n_batch` tokens per forward
/// pass. `kv_f32` keeps K/V as F32 rather than BF16, at twice the memory.
pub(crate) fn kv_init(m: &ModelData, n_ctx: usize, n_batch: usize, kv_f32: bool) -> Result<KvCache> {
    let h = &m.hparams;
    if n_ctx == 0 {
        bail!(InvalidArgument, "the context size must be positive");
    }
    if n_ctx > MAX_CTX {
        bail!(
            InvalidArgument,
            "a {n_ctx}-token context is too large (at most {MAX_CTX})"
        );
    }
    let oom = || format!("out of memory allocating a {n_ctx}-token context");
    // Sizes that grow with the context, which the caller chooses.
    let ctx_elems = |a: usize, b: usize| a.checked_mul(b).ok_or_else(|| err!(OutOfMemory, "{}", oom()));
    let nb = n_batch.max(1);
    let n_layer = h.n_layer;
    let mut c = KvCache {
        n_ctx,
        n_batch: nb,
        kv_bf16: !kv_f32,
        max_rewind: usize::MAX,
        kv_dim_per_layer: vec![0; n_layer],
        kv_buf: vec![None; n_layer],
        ..KvCache::default()
    };
    let (mut max_q_dim, mut max_kv_dim, mut max_ff, mut max_n_head, mut max_ff_exp, mut max_part) = (0, 0, 0, 1, 0, 0);
    let mut gated = false;
    for (li, l) in m.layers.iter().enumerate() {
        let kv_dim = l.n_head_kv * l.head_dim;
        c.kv_dim_per_layer[li] = kv_dim;
        max_q_dim = max_q_dim.max(l.n_head * l.head_dim);
        max_kv_dim = max_kv_dim.max(kv_dim);
        max_ff = max_ff.max(l.n_ff);
        max_n_head = max_n_head.max(l.n_head);
        max_ff_exp = max_ff_exp.max(l.n_ff_exp);
        max_part = max_part.max(l.n_head * kernels::attn_part(l.head_dim));
        gated |= l.attn_gated;
    }

    let new_vec = |n: usize| -> Result<KvVec> {
        Ok(if kv_f32 {
            KvVec::F32(zeros(n, oom)?)
        } else {
            KvVec::Bf16(zeros(n, oom)?)
        })
    };
    let esize = if kv_f32 { 4 } else { 2 };
    // One buffer per cache-owning layer; layers that reuse another layer's
    // cache, and every layer of a bidirectional model, share.
    if !h.causal {
        let n = ctx_elems(n_ctx, max_kv_dim)?;
        c.bufs = vec![KvBuffer {
            k: new_vec(n)?,
            v: new_vec(n)?,
            rows: n_ctx,
            window: 0,
        }];
        c.bytes = 2 * n * esize;
        // Every layer rewrites buffer 0 before it attends.
        for (li, l) in m.layers.iter().enumerate() {
            if l.kv_layer.is_some() {
                c.kv_buf[li] = Some(0);
            }
        }
    } else {
        // A sliding layer's buffer becomes a ring when that saves memory and
        // every layer reading it slides the same way.
        let ring = (h.sliding_window + nb).div_ceil(ATTN_BLOCK) * ATTN_BLOCK;
        let mut can_ring = vec![h.sliding_window > 0 && ring < n_ctx; n_layer];
        for l in &m.layers {
            if !l.sliding_window
                && let Some(k) = l.kv_layer
            {
                can_ring[k] = false;
            }
        }
        c.bufs = (0..n_layer).map(|_| KvBuffer::default()).collect();
        for (li, l) in m.layers.iter().enumerate() {
            c.kv_buf[li] = l.kv_layer;
            if l.kv_layer != Some(li) {
                continue;
            }
            let rows = if can_ring[li] { ring } else { n_ctx };
            let window = if can_ring[li] { h.sliding_window } else { 0 };
            if window > 0 {
                c.max_rewind = c.max_rewind.min(rows - window);
            }
            let n = ctx_elems(rows, c.kv_dim_per_layer[li])?;
            c.bufs[li] = KvBuffer {
                k: new_vec(n)?,
                v: new_vec(n)?,
                rows,
                window,
            };
            c.bytes += 2 * n * esize;
        }
    }

    let ple_dim = h.ple_dim;
    let ple_total = ple_dim * n_layer;
    let s = &mut c.s;
    s.x = zeros(nb * h.n_embd, oom)?;
    s.normed = zeros(nb * h.n_embd, oom)?;
    s.q = zeros(nb * max_q_dim, oom)?;
    s.k_cur = zeros(nb * max_kv_dim, oom)?;
    s.v_cur = zeros(nb * max_kv_dim, oom)?;
    s.attn = zeros(nb * max_q_dim, oom)?;
    s.gate = zeros(nb * max_ff, oom)?;
    s.up = zeros(nb * max_ff, oom)?;
    // Scores are held for every query head of a KV head at once so the K/V
    // cache is walked once per KV head rather than once per query head; with
    // them, a slot holds two partials. Attention runs in parallel, one slot
    // per chunk of its work, up to one per thread.
    c.n_head_max = max_n_head;
    c.attn_part_max = max_part;
    c.attn_slot = max_n_head * ATTN_BLOCK + 2 * max_part;
    c.n_attn_slots = pool::num_threads().clamp(1, 64);
    s.scores = zeros(c.attn_slot * c.n_attn_slots, oom)?;
    c.n_attn_blocks = n_ctx.div_ceil(ATTN_BLOCK) + 1;
    s.attn_part = zeros(ctx_elems(max_part, c.n_attn_blocks)?, oom)?;
    s.ple_gate = zeros(nb * ple_dim.max(1), oom)?;
    s.ple_contrib = zeros(nb * h.n_embd, oom)?;
    s.ple_inputs = zeros(nb * ple_total.max(1), oom)?;
    // The projection half of the per-layer inputs needs its own buffer only
    // when there is a token half to combine it with.
    s.ple_proj = if m.per_layer_token_embd.is_some() {
        zeros(nb * ple_total.max(1), oom)?
    } else {
        Vec::new()
    };
    // Q4 block sums: one run of (cols/32) floats per batch slot, sized to the
    // widest gemv input we will ever see (an expert's down projection reads
    // n_ff_exp values, a Gated DeltaNet's output projection n_v * d_v).
    let v_dim = h.ssm_n_v * h.ssm_d_v;
    let max_cols = [h.n_embd, max_q_dim, max_ff, ple_dim, max_ff_exp, v_dim, 32]
        .into_iter()
        .max()
        .unwrap();
    s.xsum = zeros(nb * (max_cols / 32 + 1), oom)?;
    if gated {
        s.q_gate = zeros(nb * 2 * max_q_dim, oom)?;
        s.attn_gate = zeros(nb * max_q_dim, oom)?;
    }
    if h.n_layer_rs > 0 {
        let conv_dim = 2 * h.ssm_n_k * h.ssm_d_k + v_dim;
        let hist = h.ssm_d_conv - 1;
        let n_conv = h.n_layer_rs * hist * conv_dim;
        let n_ssm = h.n_layer_rs * v_dim * h.ssm_d_k;
        c.conv_state = zeros(n_conv, oom)?;
        c.ssm_state = zeros(n_ssm, oom)?;
        c.conv_saved = zeros(n_conv, oom)?;
        c.ssm_saved = zeros(n_ssm, oom)?;
        c.bytes += 2 * (n_conv + n_ssm) * 4;
        let s = &mut c.s;
        s.dn_conv = zeros((hist + nb) * conv_dim, oom)?;
        s.dn_qkv = zeros(nb * conv_dim, oom)?;
        s.dn_z = zeros(nb * v_dim, oom)?;
        s.dn_gb = zeros(nb * 2 * h.ssm_n_v, oom)?;
        s.dn_out = zeros(nb * v_dim, oom)?;
    }
    let s = &mut c.s;
    if max_ff_exp > 0 {
        let k = h.n_expert_used;
        s.moe_in = zeros(nb * h.n_embd, oom)?;
        s.moe_out = zeros(nb * h.n_embd, oom)?;
        s.moe_rows = zeros(nb * h.n_embd, oom)?;
        s.moe_down = zeros(nb * h.n_embd, oom)?;
        s.moe_gate_up = zeros(nb * 2 * max_ff_exp, oom)?;
        s.moe_router = zeros(h.n_embd, oom)?;
        s.moe_probs = zeros(h.n_expert, oom)?;
        s.moe_weight = zeros(nb * k, oom)?;
        s.moe_expert = zeros(nb * k, oom)?;
        s.moe_pairs = zeros(nb * k, oom)?;
    }
    if h.n_embd_target > 0 {
        s.draft_in = zeros(2 * h.n_embd_target, oom)?;
        s.draft_logits = zeros(h.n_vocab, oom)?;
    }
    Ok(c)
}

/// Move cur_pos back to `pos` (<= cur_pos), keeping the K/V below it. A ring
/// buffer may have overwritten keys that position needs, when it is more
/// than max_rewind below the highest position written; then this returns
/// false and changes nothing, and the caller goes back to 0 instead (which
/// always succeeds). Recurrent state is not touched: see state_restore.
pub(crate) fn kv_rewind(c: &mut KvCache, pos: usize) -> bool {
    if pos > c.cur_pos {
        return false;
    }
    if pos > 0 && c.kv_written - pos > c.max_rewind {
        return false;
    }
    c.cur_pos = pos;
    true
}

/// Copy the recurrent state, with cur_pos. Nothing for other models.
pub(crate) fn state_save(c: &mut KvCache) {
    if c.conv_state.is_empty() {
        return;
    }
    c.conv_saved.copy_from_slice(&c.conv_state);
    c.ssm_saved.copy_from_slice(&c.ssm_state);
    c.saved_pos = Some(c.cur_pos);
}

/// Put the saved recurrent state and cur_pos back; false when nothing was
/// saved.
pub(crate) fn state_restore(c: &mut KvCache) -> bool {
    let Some(pos) = c.saved_pos else { return false };
    if c.conv_state.is_empty() {
        return false;
    }
    c.conv_state.copy_from_slice(&c.conv_saved);
    c.ssm_state.copy_from_slice(&c.ssm_saved);
    c.cur_pos = pos;
    true
}
