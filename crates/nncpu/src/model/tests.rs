//! Model tests on tiny synthetic models: random F32 weights in a GGUF in the
//! temp directory, loaded as Q8_0.

use super::kv::{KvBuffer, KvVec, MAX_CTX};
use super::*;
use crate::error::ErrorKind;
use crate::gguf::tests::{GgufWriter, temp_path};
use std::path::{Path, PathBuf};

pub(crate) const TINY_EMBD: usize = 64;
pub(crate) const TINY_VOCAB: usize = 30;
pub(crate) const TINY_TOKENS: [i32; 11] = [3, 7, 1, 29, 5, 12, 0, 18, 26, 9, 4];

pub(crate) fn tiny_token(i: usize) -> i32 {
    TINY_TOKENS[(i * 7 + i / 11) % 11]
}

/// Deterministic normally distributed values (xorshift64* and Box-Muller).
pub(crate) struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn normal(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    }

    pub fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.normal() * scale).collect()
    }

    /// Norm gammas around 1.
    pub fn gamma(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| 1.0 + 0.1 * self.normal()).collect()
    }
}

/// A 30-token byte-level vocabulary: a-z, two merges, two control tokens.
fn tiny_vocab(w: &mut GgufWriter, pre: &str) {
    let mut toks: Vec<String> = ('a'..='z').map(String::from).collect();
    toks.extend(["ab", "cd", "<|im_start|>", "<|im_end|>"].map(String::from));
    let refs: Vec<&str> = toks.iter().map(String::as_str).collect();
    let mut types = vec![1; 28];
    types.extend([3, 3]);
    w.str("tokenizer.ggml.model", "gpt2")
        .str("tokenizer.ggml.pre", pre)
        .strs("tokenizer.ggml.tokens", &refs)
        .i32s("tokenizer.ggml.token_type", &types)
        .strs("tokenizer.ggml.merges", &["a b", "c d"]);
}

/// Qwen3.5 in miniature: three Gated DeltaNet layers, then one gated
/// attention layer, and a trailing multi-token-prediction block that the
/// loader must skip. `extra` names a tensor to add that nncpu has no use for.
pub(crate) fn write_tiny_qwen35(name: &str, extra: Option<&str>) -> PathBuf {
    const N_FF: usize = 96;
    const N_LAYER: u32 = 4;
    const HEAD_DIM: usize = 32;
    const N_HEAD: usize = 2;
    const N_KV: usize = 1;
    const D_K: usize = 32;
    const N_K: usize = 2;
    const N_V: usize = 4;
    const D_CONV: usize = 4;
    const V_DIM: usize = N_V * D_K;
    const CONV_DIM: usize = 2 * N_K * D_K + V_DIM;
    let e = TINY_EMBD;
    let mut r = Rng::new(35);
    let mut g = GgufWriter::default();
    g.str("general.architecture", "qwen35")
        .u32("qwen35.block_count", N_LAYER + 1)
        .u32("qwen35.nextn_predict_layers", 1)
        .u32("qwen35.context_length", 256)
        .u32("qwen35.embedding_length", e as u32)
        .u32("qwen35.feed_forward_length", N_FF as u32)
        .u32("qwen35.attention.head_count", N_HEAD as u32)
        .u32("qwen35.attention.head_count_kv", N_KV as u32)
        .u32("qwen35.attention.key_length", HEAD_DIM as u32)
        .f32("qwen35.attention.layer_norm_rms_epsilon", 1e-6)
        .f32("qwen35.rope.freq_base", 10000.0)
        .u32("qwen35.rope.dimension_count", 16)
        .u32("qwen35.ssm.conv_kernel", D_CONV as u32)
        .u32("qwen35.ssm.state_size", D_K as u32)
        .u32("qwen35.ssm.group_count", N_K as u32)
        .u32("qwen35.ssm.time_step_rank", N_V as u32)
        .u32("qwen35.ssm.inner_size", V_DIM as u32)
        .bools("qwen35.attention.recurrent_layers", &[true, true, true, false, false]);
    tiny_vocab(&mut g, "qwen35");

    let u = |n: usize| n as u64;
    g.tensor_f32("token_embd.weight", &[u(e), u(TINY_VOCAB)], &r.vec(e * TINY_VOCAB, 0.5));
    g.tensor_f32("output_norm.weight", &[u(e)], &r.gamma(e));
    for li in 0..N_LAYER {
        let p = |s: &str| format!("blk.{li}.{s}");
        g.tensor_f32(&p("attn_norm.weight"), &[u(e)], &r.gamma(e));
        g.tensor_f32(&p("post_attention_norm.weight"), &[u(e)], &r.gamma(e));
        g.tensor_f32(&p("ffn_gate.weight"), &[u(e), u(N_FF)], &r.vec(e * N_FF, 0.15));
        g.tensor_f32(&p("ffn_up.weight"), &[u(e), u(N_FF)], &r.vec(e * N_FF, 0.15));
        g.tensor_f32(&p("ffn_down.weight"), &[u(N_FF), u(e)], &r.vec(N_FF * e, 0.1));
        if li < 3 {
            let a: Vec<f32> = (0..N_V).map(|_| -(0.5 * r.normal()).exp()).collect();
            g.tensor_f32(&p("attn_qkv.weight"), &[u(e), u(CONV_DIM)], &r.vec(e * CONV_DIM, 0.15));
            g.tensor_f32(&p("attn_gate.weight"), &[u(e), u(V_DIM)], &r.vec(e * V_DIM, 0.15));
            g.tensor_f32(&p("ssm_alpha.weight"), &[u(e), u(N_V)], &r.vec(e * N_V, 0.1));
            g.tensor_f32(&p("ssm_beta.weight"), &[u(e), u(N_V)], &r.vec(e * N_V, 0.1));
            g.tensor_f32(
                &p("ssm_conv1d.weight"),
                &[u(D_CONV), u(CONV_DIM)],
                &r.vec(D_CONV * CONV_DIM, 0.4),
            );
            g.tensor_f32(&p("ssm_dt.bias"), &[u(N_V)], &r.vec(N_V, 0.5));
            g.tensor_f32(&p("ssm_a"), &[u(N_V)], &a);
            g.tensor_f32(&p("ssm_norm.weight"), &[u(D_K)], &r.gamma(D_K));
            g.tensor_f32(&p("ssm_out.weight"), &[u(V_DIM), u(e)], &r.vec(V_DIM * e, 0.1));
        } else {
            let q_rows = 2 * N_HEAD * HEAD_DIM;
            g.tensor_f32(&p("attn_q.weight"), &[u(e), u(q_rows)], &r.vec(e * q_rows, 0.15));
            g.tensor_f32(
                &p("attn_k.weight"),
                &[u(e), u(N_KV * HEAD_DIM)],
                &r.vec(e * N_KV * HEAD_DIM, 0.15),
            );
            g.tensor_f32(
                &p("attn_v.weight"),
                &[u(e), u(N_KV * HEAD_DIM)],
                &r.vec(e * N_KV * HEAD_DIM, 0.15),
            );
            g.tensor_f32(&p("attn_q_norm.weight"), &[u(HEAD_DIM)], &r.gamma(HEAD_DIM));
            g.tensor_f32(&p("attn_k_norm.weight"), &[u(HEAD_DIM)], &r.gamma(HEAD_DIM));
            let o = N_HEAD * HEAD_DIM;
            g.tensor_f32(&p("attn_output.weight"), &[u(o), u(e)], &r.vec(o * e, 0.1));
        }
    }
    g.tensor_f32("blk.4.nextn.eh_proj.weight", &[u(2 * e), u(e)], &r.vec(2 * e * e, 0.1));
    if let Some(name) = extra {
        g.tensor_f32(name, &[1], &[1.0]);
    }
    let path = temp_path(name);
    g.write(&path);
    path
}

/// A dense Qwen3 in miniature with two attention layers, the first sliding
/// over `window` keys, the second global.
pub(crate) fn write_tiny_qwen3(name: &str, window: u32) -> PathBuf {
    const N_FF: usize = 96;
    const HEAD_DIM: usize = 32;
    const N_HEAD: usize = 4;
    const N_KV: usize = 2;
    let e = TINY_EMBD;
    let mut r = Rng::new(3);
    let mut g = GgufWriter::default();
    g.str("general.architecture", "qwen3")
        .u32("qwen3.block_count", 2)
        .u32("qwen3.context_length", 4096)
        .u32("qwen3.embedding_length", e as u32)
        .u32("qwen3.feed_forward_length", N_FF as u32)
        .u32("qwen3.attention.head_count", N_HEAD as u32)
        .u32("qwen3.attention.head_count_kv", N_KV as u32)
        .u32("qwen3.attention.key_length", HEAD_DIM as u32)
        .f32("qwen3.attention.layer_norm_rms_epsilon", 1e-6)
        .f32("qwen3.rope.freq_base", 10000.0)
        .u32("qwen3.attention.sliding_window", window)
        .bools("qwen3.attention.sliding_window_pattern", &[true, false]);
    tiny_vocab(&mut g, "qwen2");

    let u = |n: usize| n as u64;
    g.tensor_f32("token_embd.weight", &[u(e), u(TINY_VOCAB)], &r.vec(e * TINY_VOCAB, 0.5));
    g.tensor_f32("output_norm.weight", &[u(e)], &r.gamma(e));
    for li in 0..2 {
        let p = |s: &str| format!("blk.{li}.{s}");
        g.tensor_f32(&p("attn_norm.weight"), &[u(e)], &r.gamma(e));
        g.tensor_f32(&p("ffn_norm.weight"), &[u(e)], &r.gamma(e));
        g.tensor_f32(&p("ffn_gate.weight"), &[u(e), u(N_FF)], &r.vec(e * N_FF, 0.15));
        g.tensor_f32(&p("ffn_up.weight"), &[u(e), u(N_FF)], &r.vec(e * N_FF, 0.15));
        g.tensor_f32(&p("ffn_down.weight"), &[u(N_FF), u(e)], &r.vec(N_FF * e, 0.1));
        let (q, kv) = (N_HEAD * HEAD_DIM, N_KV * HEAD_DIM);
        g.tensor_f32(&p("attn_q.weight"), &[u(e), u(q)], &r.vec(e * q, 0.3));
        g.tensor_f32(&p("attn_k.weight"), &[u(e), u(kv)], &r.vec(e * kv, 0.3));
        g.tensor_f32(&p("attn_v.weight"), &[u(e), u(kv)], &r.vec(e * kv, 0.3));
        g.tensor_f32(&p("attn_q_norm.weight"), &[u(HEAD_DIM)], &r.gamma(HEAD_DIM));
        g.tensor_f32(&p("attn_k_norm.weight"), &[u(HEAD_DIM)], &r.gamma(HEAD_DIM));
        g.tensor_f32(&p("attn_output.weight"), &[u(q), u(e)], &r.vec(q * e, 0.1));
    }
    let path = temp_path(name);
    g.write(&path);
    path
}

pub(crate) fn load_tiny(path: &Path) -> ModelData {
    load(path, WeightFormat::Q8_0, None).unwrap()
}

fn kv_len(v: &KvVec) -> usize {
    match v {
        KvVec::Bf16(v) => v.len(),
        KvVec::F32(v) => v.len(),
    }
}

fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// Feed tokens [from, to) in batches of up to `batch`; true if each call
/// leaves the same final hidden states in both caches.
fn forward_both(m: &ModelData, a: &mut KvCache, b: &mut KvCache, from: usize, to: usize, batch: usize) -> bool {
    let mut p = from;
    while p < to {
        let n = batch.min(to - p);
        let ids: Vec<i32> = (p..p + n).map(tiny_token).collect();
        model_forward(m, a, &ids, p).unwrap();
        model_forward(m, b, &ids, p).unwrap();
        if !same_bits(&a.s.x[..n * TINY_EMBD], &b.s.x[..n * TINY_EMBD]) {
            return false;
        }
        p += n;
    }
    true
}

/// A sliding layer's ring buffer gives exactly what a full buffer does,
/// through many wraps and a step back, and refuses to go back further than
/// it holds.
#[test]
fn sliding_ring_matches_full() {
    let path = write_tiny_qwen3("ring.gguf", 40);
    {
        let m = load_tiny(&path);
        assert!(m.layers[0].sliding_window && !m.layers[1].sliding_window);
        let mut ring = kv_init(&m, 1200, 16, false).unwrap();
        let mut full = kv_init(&m, 1200, 16, false).unwrap();
        assert_eq!(
            (ring.bufs[0].rows, ring.bufs[0].window, ring.bufs[1].rows),
            (256, 40, 1200)
        );
        assert_eq!(ring.max_rewind, 216);
        // The same cache with the sliding layer kept in full.
        let n = 1200 * 64;
        full.bufs[0] = KvBuffer {
            k: KvVec::Bf16(vec![0; n]),
            v: KvVec::Bf16(vec![0; n]),
            rows: 1200,
            window: 0,
        };
        full.bytes += 2 * 2 * (n - kv_len(&ring.bufs[0].k));
        full.max_rewind = usize::MAX;
        assert!(ring.bytes < full.bytes);

        assert!(forward_both(&m, &mut ring, &mut full, 0, 700, 16));
        assert!(forward_both(&m, &mut ring, &mut full, 700, 720, 1));
        assert!(kv_rewind(&mut ring, 705) && kv_rewind(&mut full, 705));
        assert!(forward_both(&m, &mut ring, &mut full, 705, 760, 16));
        assert!(!kv_rewind(&mut ring, 760 - 217) && ring.cur_pos == 760);
        assert!(kv_rewind(&mut ring, 760 - 216));
        assert!(kv_rewind(&mut ring, 0) && kv_rewind(&mut full, 0));
        assert!(forward_both(&m, &mut ring, &mut full, 0, 300, 16));
    }
    std::fs::remove_file(&path).unwrap();
}

/// Decode attention over a long span is shared out by (KV head, block), yet
/// each token gets the bits a batch gives it, with F32 or BF16 K/V.
#[test]
fn attn_decode_matches_batch() {
    let path = write_tiny_qwen3("decode.gguf", 0);
    {
        let m = load_tiny(&path);
        for f32_kv in [false, true] {
            let mut a = kv_init(&m, 1200, 16, f32_kv).unwrap();
            let mut b = kv_init(&m, 1200, 16, f32_kv).unwrap();
            assert!(forward_both(&m, &mut a, &mut b, 0, 1100, 16));
            let ids = [tiny_token(1100), tiny_token(1101), tiny_token(1102)];
            model_forward(&m, &mut a, &ids, 1100).unwrap();
            let batch = a.s.x[..3 * TINY_EMBD].to_vec();
            for t in 0..3 {
                model_forward(&m, &mut b, &ids[t..t + 1], 1100 + t).unwrap();
                assert!(
                    same_bits(&b.s.x[..TINY_EMBD], &batch[t * TINY_EMBD..(t + 1) * TINY_EMBD]),
                    "token {t}"
                );
            }
        }
    }
    std::fs::remove_file(&path).unwrap();
}

fn close(a: f32, b: f32, rel: f32) -> bool {
    (a - b).abs() <= rel * a.abs().max(b.abs()).max(1.0)
}

/// A batch through the Gated DeltaNet layers (the convolution window spanning
/// the saved history and the batch, the recurrence carried from token to
/// token) gives each token the state that one token at a time does.
#[test]
fn qwen35_batch_matches_steps() {
    let path = write_tiny_qwen35("qwen35.gguf", None);
    {
        let m = load_tiny(&path);
        assert_eq!((m.hparams.n_layer, m.hparams.n_layer_rs), (4, 3));
        assert!(m.layers[3].attn_gated && m.layers[3].rs_index.is_none());
        let mut a = kv_init(&m, 64, 16, false).unwrap();
        let mut b = kv_init(&m, 64, 16, false).unwrap();
        model_forward(&m, &mut a, &TINY_TOKENS, 0).unwrap();
        for t in 0..11 {
            model_forward(&m, &mut b, &TINY_TOKENS[t..t + 1], t).unwrap();
            for i in 0..TINY_EMBD {
                assert!(close(b.s.x[i], a.s.x[t * TINY_EMBD + i], 1e-4), "token {t} lane {i}");
            }
        }
    }
    std::fs::remove_file(&path).unwrap();
}

/// The recurrent state cannot be cut back: a saved copy puts it, and the
/// position, back exactly, and a sequence that starts again at position 0
/// starts from a clear state.
#[test]
fn qwen35_state_save_restore() {
    let path = write_tiny_qwen35("qwen35-state.gguf", None);
    {
        let m = load_tiny(&path);
        let mut c = kv_init(&m, 64, 16, false).unwrap();
        model_forward(&m, &mut c, &TINY_TOKENS[..6], 0).unwrap();
        state_save(&mut c);
        let rows = 5 * TINY_EMBD;
        model_forward(&m, &mut c, &TINY_TOKENS[6..], 6).unwrap();
        let first = c.s.x[..rows].to_vec();
        assert!(state_restore(&mut c) && c.cur_pos == 6);
        model_forward(&m, &mut c, &TINY_TOKENS[6..], 6).unwrap();
        assert!(same_bits(&first, &c.s.x[..rows]));
        c.cur_pos = 0;
        model_forward(&m, &mut c, &TINY_TOKENS, 0).unwrap();
        for i in 0..TINY_EMBD {
            assert!(close(c.s.x[10 * TINY_EMBD + i], first[4 * TINY_EMBD + i], 1e-5));
        }
    }
    std::fs::remove_file(&path).unwrap();
}

/// Files nncpu would run wrongly are refused: a Qwen model holding a tensor
/// nncpu has no use for, and the Qwen architectures it does not implement.
#[test]
fn qwen_refusals() {
    let path = write_tiny_qwen35("qwen35-extra.gguf", Some("blk.0.attn_qkv.scale"));
    let err = load(&path, WeightFormat::Q8_0, None).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::UnsupportedModel);
    assert!(err.message().contains("blk.0.attn_qkv.scale"), "{err}");
    std::fs::remove_file(&path).unwrap();
    for arch in ["qwen4exp", "qwen35moe"] {
        let mut g = GgufWriter::default();
        g.str("general.architecture", arch);
        let p = temp_path(&format!("arch-{arch}.gguf"));
        g.write(&p);
        let err = load(&p, WeightFormat::Q8_0, None).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::UnsupportedModel);
        std::fs::remove_file(&p).unwrap();
    }
}

/// Corrupt sizes are refused rather than allocated: a block count beyond
/// the file's tensors, and a context too large to address.
#[test]
fn size_refusals() {
    let mut g = GgufWriter::default();
    g.str("general.architecture", "qwen3")
        .u32("qwen3.block_count", i32::MAX as u32)
        .u32("qwen3.context_length", 4096)
        .u32("qwen3.embedding_length", TINY_EMBD as u32)
        .u32("qwen3.attention.head_count", 4);
    tiny_vocab(&mut g, "qwen2");
    let embd = vec![0.0; TINY_EMBD * TINY_VOCAB];
    g.tensor_f32("token_embd.weight", &[TINY_EMBD as u64, TINY_VOCAB as u64], &embd);
    let p = temp_path("huge-blocks.gguf");
    g.write(&p);
    let err = load(&p, WeightFormat::Q8_0, None).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::BadFormat, "{err}");
    std::fs::remove_file(&p).unwrap();

    let path = write_tiny_qwen3("huge-ctx.gguf", 0);
    {
        let m = load_tiny(&path);
        for n_ctx in [MAX_CTX + 1, usize::MAX] {
            let err = kv_init(&m, n_ctx, 16, false)
                .err()
                .expect("an oversized context was accepted");
            assert_eq!(err.kind(), ErrorKind::InvalidArgument, "{err}");
        }
    }
    std::fs::remove_file(&path).unwrap();
}

/// A Q4 matrix too wide for the stack's block sums (1025 blocks) takes the
/// heap path.
#[test]
fn wide_q4_gemv() {
    let (cols, rows) = (32 * 1025, 2);
    let n = cols * rows;
    let mut buf = AlignedBuf::zeroed(n / 2 + n / 8).unwrap();
    let bytes = buf.as_mut_slice();
    bytes[..n / 2].fill(0x11); // every q = 1
    for (i, b) in bytes[n / 2..].as_chunks_mut::<2>().0.iter_mut().enumerate() {
        // The scales are 1.0 and the biases 0.5, so every weight is 1.5.
        let v: u16 = if i < n / 32 { 0x3F80 } else { 0x3F00 };
        b.copy_from_slice(&v.to_ne_bytes());
    }
    let w = Tensor::new(DType::Q4S, &[cols, rows], buf.as_ptr());
    let x = vec![1.0f32; cols];
    let mut y = vec![0.0f32; rows];
    model_gemv(&w, &x, &mut y);
    assert_eq!(y, [1.5 * cols as f32; 2]);
}

/// Every weight format loads the same tiny model and agrees closely with
/// BF16 (the most precise layout).
#[test]
fn weight_formats_agree() {
    let path = write_tiny_qwen3("formats.gguf", 0);
    {
        let run = |fmt: WeightFormat| {
            let m = load(&path, fmt, None).unwrap();
            let mut c = kv_init(&m, 64, 16, false).unwrap();
            model_forward(&m, &mut c, &TINY_TOKENS, 0).unwrap();
            let mut logits = vec![0.0; m.hparams.n_vocab];
            model_logits(&m, model_hidden(&m, &c, 10), &mut logits);
            logits
        };
        let bf16 = run(WeightFormat::Bf16);
        let norm = bf16.iter().map(|v| v * v).sum::<f32>().sqrt();
        // Relative RMS error: about 1.4% for Q8_0 and 21% for Q4 on these
        // random weights, in line with each format's quantization step.
        for (fmt, tol) in [(WeightFormat::Q8_0, 0.05), (WeightFormat::Q4, 0.5)] {
            let l = run(fmt);
            let err = l.iter().zip(&bf16).map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt() / norm;
            assert!(err < tol, "{fmt:?}: relative error {err}");
        }
    }
    std::fs::remove_file(&path).unwrap();
}
