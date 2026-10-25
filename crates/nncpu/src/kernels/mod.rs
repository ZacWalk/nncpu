//! The neural-net kernels: activations, norms, RoPE, blocked attention, the
//! Gated DeltaNet recurrence, quantization and the gemvs, which (with
//! `pool::parallel_for` callers) are what runs on the worker pool.
//!
//! Every SIMD kernel exists twice: AVX2/FMA intrinsics ([`avx2`]) and
//! portable code ([`scalar`]) that follows the same arithmetic lane by lane.
//! A [`Backend`] picks one; the free functions use [`Backend::current`]
//! (AVX2 unless the CPU lacks it or `NNCPU_NO_SIMD` is set). Kernels the
//! original ran scalar (RoPE, soft-cap, routing, embedding rows) live here,
//! shared by both.
//!
//! Sizes come from slices. A kernel's length is that of its output (or the
//! slice its docs name); inputs must be at least that long and only their
//! first elements are read. A violated precondition panics.
//!
//! Bit-exactness: a gemv's result does not depend on the thread count, and a
//! batched gemv gives each vector the bits the single-vector gemv gives it.
//! Attention blocks merged in order give the same bits wherever they ran.

// The kernel signatures mirror llmcpu's flat argument lists.
#![allow(clippy::too_many_arguments)]

#[cfg(target_arch = "x86_64")]
mod avx2;
mod quant;
mod scalar;
#[cfg(test)]
mod tests;

pub(crate) use quant::{dequantize_to_f32, quant_block_elems, quantize_f32_to_q4, quantize_f32_to_q8_0};

use crate::pool::{self, MAX_THREADS};
use crate::runtime::bf16_to_f32;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// How RoPE pairs a head's lanes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RopeMode {
    /// Lane i with lane i + n_rot/2 (Gemma, Qwen, Falcon, ...).
    Neox,
    /// Lanes 2i and 2i + 1 (Llama, whose Q/K weights the GGUF converter
    /// permutes to suit it).
    Norm,
}

/// The K or V rows of one KV head: its lanes at offset 0 of each row, rows
/// `kv_stride` elements apart.
#[derive(Clone, Copy, Debug)]
pub(crate) enum KvRows<'a> {
    Bf16(&'a [u16]),
    F32(&'a [f32]),
}

/// Floats per head in an attention partial: the unnormalised output
/// `o[head_dim]`, then the block's max score `m`, then the sum of its
/// weights `l = sum_t exp(s_t - m)`.
pub(crate) const fn attn_part(head_dim: usize) -> usize {
    head_dim + 2
}

/// `a * b`, panicking on overflow (sizes checked before unchecked loops).
fn elems(a: usize, b: usize) -> usize {
    a.checked_mul(b).expect("kernel buffer size overflows usize")
}

/// Elements spanned by `n` runs of `len`, `stride` apart.
fn strided_len(n: usize, stride: usize, len: usize) -> usize {
    if n == 0 {
        0
    } else {
        elems(n - 1, stride)
            .checked_add(len)
            .expect("kernel buffer size overflows usize")
    }
}

fn rows_u32(rows: usize) -> u32 {
    u32::try_from(rows).expect("gemv: rows exceeds u32")
}

/// [`attn_block`]'s preconditions; shared with the AVX2 kernel, whose
/// unchecked loops rely on them.
fn check_attn(
    q: &[f32],
    q_stride: usize,
    k: KvRows<'_>,
    v: KvRows<'_>,
    kv_stride: usize,
    n_keys: usize,
    head_dim: usize,
    n_heads: usize,
    scores: &[f32],
    part: &[f32],
) {
    let kv = strided_len(n_keys, kv_stride, head_dim);
    let kv_ok = match (k, v) {
        (KvRows::F32(k), KvRows::F32(v)) => k.len() >= kv && v.len() >= kv,
        (KvRows::Bf16(k), KvRows::Bf16(v)) => k.len() >= kv && v.len() >= kv,
        _ => panic!("attn_block: K and V must have the same type"),
    };
    assert!(kv_ok, "attn_block: K/V rows too short");
    assert!(
        q.len() >= strided_len(n_heads, q_stride, head_dim),
        "attn_block: q too short"
    );
    assert!(scores.len() >= elems(n_heads, n_keys), "attn_block: scores too short");
    assert!(
        part.len() >= elems(n_heads, attn_part(head_dim)),
        "attn_block: part too short"
    );
}

/// A kernel implementation.
#[derive(Clone, Copy, Debug)]
enum Backend {
    /// Portable code, for any CPU.
    Scalar,
    /// AVX2 + FMA; the token proves the CPU has them.
    #[cfg(target_arch = "x86_64")]
    Avx2(avx2::Avx2),
}

/// Call `$f` on the backend's implementation: a method of the AVX2 token or
/// the `scalar` function of the same name and arguments.
macro_rules! by_backend {
    ($be:expr, $f:ident($($arg:expr),* $(,)?)) => {
        match $be {
            #[cfg(target_arch = "x86_64")]
            Backend::Avx2(k) => k.$f($($arg),*),
            Backend::Scalar => scalar::$f($($arg),*),
        }
    };
}

impl Backend {
    /// AVX2/FMA when [`crate::cpu::simd`] allows it, else portable.
    #[inline]
    fn current() -> Backend {
        static CURRENT: OnceLock<Backend> = OnceLock::new();
        *CURRENT.get_or_init(|| {
            #[cfg(target_arch = "x86_64")]
            if crate::cpu::simd()
                && let Some(k) = avx2::Avx2::detect()
            {
                return Backend::Avx2(k);
            }
            Backend::Scalar
        })
    }

    /// Every implementation this CPU runs, for tests to compare.
    #[cfg(test)]
    fn all() -> Vec<Backend> {
        #[allow(unused_mut)]
        let mut v = vec![Backend::Scalar];
        #[cfg(target_arch = "x86_64")]
        v.extend(avx2::Avx2::detect().map(Backend::Avx2));
        v
    }

    fn gelu_inplace(self, x: &mut [f32]) {
        by_backend!(self, gelu_inplace(x))
    }

    fn gelu_mul_inplace(self, gate: &mut [f32], up: &[f32]) {
        let up = &up[..gate.len()];
        by_backend!(self, gelu_mul_inplace(gate, up))
    }

    fn swiglu_inplace(self, gate: &mut [f32], up: &[f32]) {
        let up = &up[..gate.len()];
        by_backend!(self, swiglu_inplace(gate, up))
    }

    fn silu_mul_into(self, up: &mut [f32], gate: &[f32]) {
        let gate = &gate[..up.len()];
        by_backend!(self, silu_mul_into(up, gate))
    }

    fn mul_sigmoid(self, y: &mut [f32], g: &[f32]) {
        let g = &g[..y.len()];
        by_backend!(self, mul_sigmoid(y, g))
    }

    fn add_inplace(self, y: &mut [f32], b: &[f32]) {
        let b = &b[..y.len()];
        by_backend!(self, add_inplace(y, b))
    }

    fn mul_inplace(self, y: &mut [f32], s: &[f32]) {
        let s = &s[..y.len()];
        by_backend!(self, mul_inplace(y, s))
    }

    fn softmax_inplace(self, p: &mut [f32]) {
        by_backend!(self, softmax_inplace(p))
    }

    fn rmsnorm(self, y: &mut [f32], x: &[f32], eps: f32) {
        let x = &x[..y.len()];
        by_backend!(self, rmsnorm(y, x, eps))
    }

    fn rmsnorm_inplace(self, x: &mut [f32], eps: f32) {
        by_backend!(self, rmsnorm_inplace(x, eps))
    }

    fn rmsnorm_gamma_multi(self, y: &mut [f32], x: &[f32], dim: usize, gamma: &[f32], eps: f32) {
        if x.is_empty() || dim == 0 {
            return;
        }
        assert!(
            x.len().is_multiple_of(dim),
            "rmsnorm_gamma_multi: {} floats are not whole groups of {dim}",
            x.len()
        );
        let (y, gamma) = (&mut y[..x.len()], &gamma[..dim]);
        by_backend!(self, rmsnorm_gamma_multi(y, x, dim, gamma, eps))
    }

    fn rmsnorm_gamma_multi_inplace(self, x: &mut [f32], dim: usize, gamma: &[f32], eps: f32) {
        if x.is_empty() || dim == 0 {
            return;
        }
        assert!(
            x.len().is_multiple_of(dim),
            "rmsnorm_gamma_multi: {} floats are not whole groups of {dim}",
            x.len()
        );
        let gamma = &gamma[..dim];
        by_backend!(self, rmsnorm_gamma_multi_inplace(x, dim, gamma, eps))
    }

    fn l2norm_multi(self, x: &mut [f32], dim: usize, eps: f32, scale: f32) {
        if x.is_empty() {
            return;
        }
        assert!(
            dim > 0 && x.len().is_multiple_of(dim),
            "l2norm_multi: {} floats are not whole groups of {dim}",
            x.len()
        );
        by_backend!(self, l2norm_multi(x, dim, eps, scale))
    }

    fn dot(self, a: &[f32], b: &[f32]) -> f32 {
        let b = &b[..a.len()];
        by_backend!(self, dot(a, b))
    }

    fn bf16_to_f32_row(self, src: &[u16], dst: &mut [f32]) {
        let src = &src[..dst.len()];
        by_backend!(self, bf16_to_f32_row(src, dst))
    }

    fn block_sums(self, x: &[f32], out: &mut [f32]) {
        assert!(
            x.len().is_multiple_of(32),
            "block_sums: {} floats are not whole 32-blocks",
            x.len()
        );
        let out = &mut out[..x.len() / 32];
        by_backend!(self, block_sums(x, out))
    }

    fn conv1d_silu(self, y: &mut [f32], x: &[f32], w: &[f32], n_tok: usize, ch: usize, k: usize) {
        let y = &mut y[..elems(n_tok, ch)];
        if y.is_empty() {
            return;
        }
        if k == 0 {
            // No taps: silu(0).
            y.fill(0.0);
            return;
        }
        let (x, w) = (&x[..elems(k - 1 + n_tok, ch)], &w[..elems(k, ch)]);
        by_backend!(self, conv1d_silu(y, x, w, n_tok, ch, k))
    }

    fn delta_rule(
        self,
        s: &mut [f32],
        q: &[f32],
        k: &[f32],
        v: &[f32],
        decay: f32,
        beta: f32,
        o: &mut [f32],
        d_k: usize,
        d_v: usize,
    ) {
        let (s, v, o) = (&mut s[..elems(d_v, d_k)], &v[..d_v], &mut o[..d_v]);
        if d_k == 0 {
            // An empty state reads out zeros.
            o.fill(0.0);
            return;
        }
        let (q, k) = (&q[..d_k], &k[..d_k]);
        by_backend!(self, delta_rule(s, q, k, v, decay, beta, o, d_k, d_v))
    }

    fn attn_block(
        self,
        q: &[f32],
        q_stride: usize,
        k: KvRows<'_>,
        v: KvRows<'_>,
        kv_stride: usize,
        n_keys: usize,
        head_dim: usize,
        n_heads: usize,
        scale: f32,
        scores: &mut [f32],
        part: &mut [f32],
    ) {
        if n_keys == 0 || n_heads == 0 {
            return;
        }
        check_attn(q, q_stride, k, v, kv_stride, n_keys, head_dim, n_heads, scores, part);
        by_backend!(
            self,
            attn_block(
                q, q_stride, k, v, kv_stride, n_keys, head_dim, n_heads, scale, scores, part
            )
        )
    }

    fn attn_merge(self, acc: &mut [f32], part: &[f32], head_dim: usize, n_heads: usize) {
        let n = elems(n_heads, attn_part(head_dim));
        let (acc, part) = (&mut acc[..n], &part[..n]);
        by_backend!(self, attn_merge(acc, part, head_dim, n_heads))
    }

    // ---- gemv: y[r] = sum_k W[r, k] x[k], the row axis split across the pool ----

    fn gemv_bf16(self, w: &[u16], x: &[f32], y: &mut [f32], rows: usize, cols: usize) {
        if rows == 0 || cols == 0 {
            return;
        }
        let (w, x, y) = (&w[..elems(rows, cols)], &x[..cols], &mut y[..rows]);
        let units = rows_u32(rows);
        if !cols.is_multiple_of(8) {
            return gemv_bf16_f64(w, x, y, cols);
        }
        if rows.is_multiple_of(4) {
            // The 4-row kernel, threaded when the gemv is big enough to pay
            // for dispatch: rows >= 256 catches every significant weight
            // (Q/K/V/O/gate/up/down, lm_head) but keeps tiny ones (PLE on a
            // small ple_dim) on the single-threaded path.
            let threaded = pool::num_threads() > 1 && rows >= 256 && cols >= 256;
            pool::for_rows(threaded, units / 4, 4, y, rows, 1, |span| {
                by_backend!(self, gemv_bf16_quads(w, x, 0, cols, span))
            });
        } else {
            pool::for_rows(false, units, 1, y, rows, 1, |span| {
                by_backend!(self, gemv_bf16_rows(w, x, cols, span))
            });
        }
    }

    /// The argmax of the BF16 gemv without materialising it: 4 rows at a
    /// time into a running best, each thread its own (lm_head's 262144 rows
    /// make this the highest-payoff parallel site). Ties go to the lower
    /// row; all-NaN rows give 0.
    fn gemv_bf16_argmax(self, w: &[u16], x: &[f32], rows: usize, cols: usize) -> usize {
        assert!(rows > 0, "gemv_bf16_argmax: no rows");
        if cols == 0 {
            return 0;
        }
        let (w, x) = (&w[..elems(rows, cols)], &x[..cols]);
        if !rows.is_multiple_of(4) || !cols.is_multiple_of(8) {
            let mut y = vec![0.0; rows];
            self.gemv_bf16(w, x, &mut y, rows, cols);
            let mut best = 0;
            for r in 1..rows {
                if y[r] > y[best] {
                    best = r;
                }
            }
            return best;
        }
        let quads = rows_u32(rows) / 4;
        if !(pool::num_threads() > 1 && rows >= 256 && cols >= 256) {
            return by_backend!(self, gemv_bf16_argmax_quads(w, x, cols, 0..quads as usize)).map_or(0, |(_, r)| r);
        }
        // Each thread's (value, row), packed; every slot starts empty, so
        // the reduction is right whatever thread count the dispatch has.
        const EMPTY: u64 = u64::MAX;
        let best: [AtomicU64; MAX_THREADS] = [const { AtomicU64::new(EMPTY) }; MAX_THREADS];
        pool::dispatch(|tid, nth| {
            let (q0, q1) = pool::split_range(quads, tid, nth);
            if let Some((v, r)) = by_backend!(self, gemv_bf16_argmax_quads(w, x, cols, q0 as usize..q1 as usize)) {
                best[tid].store((u64::from(v.to_bits()) << 32) | r as u64, Relaxed);
            }
        });
        let mut top: Option<(f32, usize)> = None;
        for b in &best {
            let b = b.load(Relaxed);
            let (v, r) = (f32::from_bits((b >> 32) as u32), (b & 0xffff_ffff) as usize);
            if b != EMPTY && top.is_none_or(|(tv, _)| v > tv) {
                top = Some((v, r));
            }
        }
        top.map_or(0, |(_, r)| r)
    }

    fn gemv_q8_0(self, qs: &[i8], scales: &[u16], x: &[f32], y: &mut [f32], rows: usize, cols: usize) {
        if rows == 0 || cols == 0 {
            return;
        }
        assert!(
            cols.is_multiple_of(32),
            "gemv_q8_0: cols {cols} is not a multiple of 32"
        );
        let (qs, scales) = (&qs[..elems(rows, cols)], &scales[..elems(rows, cols / 32)]);
        let (x, y) = (&x[..cols], &mut y[..rows]);
        let threaded = pool::num_threads() > 1 && rows >= 256 && cols >= 256;
        pool::for_rows(threaded, rows_u32(rows), 1, y, rows, 1, |span| {
            by_backend!(self, gemv_q8_0_rows(qs, scales, x, 0, cols, span))
        });
    }

    fn gemv_q4(
        self,
        qs: &[u8],
        scales: &[u16],
        biases: &[u16],
        x: &[f32],
        xsum: &[f32],
        y: &mut [f32],
        rows: usize,
        cols: usize,
    ) {
        if rows == 0 || cols == 0 {
            return;
        }
        assert!(cols.is_multiple_of(32), "gemv_q4: cols {cols} is not a multiple of 32");
        let nb = elems(rows, cols / 32);
        let (qs, scales, biases) = (&qs[..elems(rows, cols / 2)], &scales[..nb], &biases[..nb]);
        let (x, xsum, y) = (&x[..cols], &xsum[..cols / 32], &mut y[..rows]);
        let threaded = pool::num_threads() > 1 && rows >= 256 && cols >= 256;
        pool::for_rows(threaded, rows_u32(rows), 1, y, rows, 1, |span| {
            by_backend!(self, gemv_q4_rows(qs, scales, biases, x, 0, xsum, 0, cols, span))
        });
    }

    // Batched (prefill): one pass over the weights serves the whole batch.
    // Rows are the outer loop (each is fetched from DRAM once) and the batch
    // the inner (hitting that row in L1); threads split the row axis as the
    // single-vector paths do, and run the quantised formats three tokens per
    // kernel call. Threaded from 64 rows (of quads, for BF16).

    fn gemv_bf16_batch(
        self,
        w: &[u16],
        x: &[f32],
        x_stride: usize,
        y: &mut [f32],
        y_stride: usize,
        rows: usize,
        cols: usize,
        n_batch: usize,
    ) {
        if rows == 0 || cols == 0 || n_batch == 0 {
            return;
        }
        if n_batch == 1 {
            return self.gemv_bf16(w, x, y, rows, cols);
        }
        assert!(y_stride >= rows, "gemv batch: y_stride {y_stride} < rows {rows}");
        // The 4-row kernel is the batch's unit of work; what it can't
        // express falls back to repeating the single-vector gemv.
        if !rows.is_multiple_of(4) || !cols.is_multiple_of(8) {
            for n in 0..n_batch {
                self.gemv_bf16(w, &x[n * x_stride..], &mut y[n * y_stride..], rows, cols);
            }
            return;
        }
        let (w, x) = (&w[..elems(rows, cols)], &x[..strided_len(n_batch, x_stride, cols)]);
        let threaded = pool::num_threads() > 1 && rows / 4 >= 64 && cols >= 256;
        pool::for_rows(threaded, rows_u32(rows) / 4, 4, y, y_stride, n_batch, |span| {
            by_backend!(self, gemv_bf16_quads(w, x, x_stride, cols, span))
        });
    }

    fn gemv_q8_0_batch(
        self,
        qs: &[i8],
        scales: &[u16],
        x: &[f32],
        x_stride: usize,
        y: &mut [f32],
        y_stride: usize,
        rows: usize,
        cols: usize,
        n_batch: usize,
    ) {
        if rows == 0 || cols == 0 || n_batch == 0 {
            return;
        }
        if n_batch == 1 {
            return self.gemv_q8_0(qs, scales, x, y, rows, cols);
        }
        assert!(
            cols.is_multiple_of(32),
            "gemv_q8_0: cols {cols} is not a multiple of 32"
        );
        assert!(y_stride >= rows, "gemv batch: y_stride {y_stride} < rows {rows}");
        let (qs, scales) = (&qs[..elems(rows, cols)], &scales[..elems(rows, cols / 32)]);
        let x = &x[..strided_len(n_batch, x_stride, cols)];
        let threaded = pool::num_threads() > 1 && rows >= 64 && cols >= 256;
        pool::for_rows(threaded, rows_u32(rows), 1, y, y_stride, n_batch, |span| {
            by_backend!(self, gemv_q8_0_rows(qs, scales, x, x_stride, cols, span))
        });
    }

    fn gemv_q4_batch(
        self,
        qs: &[u8],
        scales: &[u16],
        biases: &[u16],
        x: &[f32],
        x_stride: usize,
        xsum: &[f32],
        xsum_stride: usize,
        y: &mut [f32],
        y_stride: usize,
        rows: usize,
        cols: usize,
        n_batch: usize,
    ) {
        if rows == 0 || cols == 0 || n_batch == 0 {
            return;
        }
        if n_batch == 1 {
            return self.gemv_q4(qs, scales, biases, x, xsum, y, rows, cols);
        }
        assert!(cols.is_multiple_of(32), "gemv_q4: cols {cols} is not a multiple of 32");
        assert!(y_stride >= rows, "gemv batch: y_stride {y_stride} < rows {rows}");
        let nb = elems(rows, cols / 32);
        let (qs, scales, biases) = (&qs[..elems(rows, cols / 2)], &scales[..nb], &biases[..nb]);
        let x = &x[..strided_len(n_batch, x_stride, cols)];
        let xsum = &xsum[..strided_len(n_batch, xsum_stride, cols / 32)];
        let threaded = pool::num_threads() > 1 && rows >= 64 && cols >= 256;
        pool::for_rows(threaded, rows_u32(rows), 1, y, y_stride, n_batch, |span| {
            by_backend!(
                self,
                gemv_q4_rows(qs, scales, biases, x, x_stride, xsum, xsum_stride, cols, span)
            )
        });
    }
}

/// The BF16 gemv for cols that are not whole 8-lane tiles: f64 sums.
fn gemv_bf16_f64(w: &[u16], x: &[f32], y: &mut [f32], cols: usize) {
    for (y, row) in y.iter_mut().zip(w.chunks_exact(cols)) {
        let mut s = 0.0;
        for (&w, &x) in row.iter().zip(x) {
            s += f64::from(bf16_to_f32(w)) * f64::from(x);
        }
        *y = s as f32;
    }
}

// ---- the kernel API ------------------------------------------------------------------

/// x = gelu(x), the tanh form.
pub(crate) fn gelu_inplace(x: &mut [f32]) {
    Backend::current().gelu_inplace(x)
}

/// gate[i] = gelu(gate[i]) * up[i]: the fused Gemma gated-MLP step.
pub(crate) fn gelu_mul_inplace(gate: &mut [f32], up: &[f32]) {
    Backend::current().gelu_mul_inplace(gate, up)
}

/// gate[i] = silu(gate[i]) * up[i], silu(x) = x * sigmoid(x).
pub(crate) fn swiglu_inplace(gate: &mut [f32], up: &[f32]) {
    Backend::current().swiglu_inplace(gate, up)
}

/// up[i] = silu(gate[i]) * up[i] (the Gated DeltaNet output gate).
pub(crate) fn silu_mul_into(up: &mut [f32], gate: &[f32]) {
    Backend::current().silu_mul_into(up, gate)
}

/// y[i] *= sigmoid(g[i]) (Qwen3.5's gated attention output).
pub(crate) fn mul_sigmoid(y: &mut [f32], g: &[f32]) {
    Backend::current().mul_sigmoid(y, g)
}

pub(crate) fn add_inplace(y: &mut [f32], b: &[f32]) {
    Backend::current().add_inplace(y, b)
}

pub(crate) fn mul_inplace(y: &mut [f32], s: &[f32]) {
    Backend::current().mul_inplace(y, s)
}

/// Numerically stable softmax in place; -inf entries (masked) become 0, and
/// a fully masked row all zeros.
pub(crate) fn softmax_inplace(p: &mut [f32]) {
    Backend::current().softmax_inplace(p)
}

/// x = tanh(x / cap) * cap (Gemma's logit soft-cap).
pub(crate) fn softcap_inplace(x: &mut [f32], cap: f32) {
    let inv = 1.0 / cap;
    for v in x {
        *v = (*v * inv).tanh() * cap;
    }
}

/// Gemma 4's expert choice: the `k` largest of the probabilities `p` (ties
/// to the lower index) into `idx[..k]`, and their probabilities,
/// renormalised to sum to 1, into `w[..k]`. Requires 0 < k <= p.len().
pub(crate) fn moe_top_k(p: &[f32], k: usize, idx: &mut [i32], w: &mut [f32]) {
    let n = p.len();
    assert!(0 < k && k <= n && n <= i32::MAX as usize, "moe_top_k: k = {k} of {n}");
    let (idx, w) = (&mut idx[..k], &mut w[..k]);
    // k is small (8 of 128 for Gemma 4), so k passes of a linear scan beat
    // sorting the whole distribution.
    let mut sum = 0.0f32;
    for j in 0..k {
        let mut best: Option<usize> = None;
        for e in 0..n {
            if !idx[..j].contains(&(e as i32)) && best.is_none_or(|b| p[e] > p[b]) {
                best = Some(e);
            }
        }
        let b = best.expect("j < k <= n leaves an expert untaken");
        idx[j] = b as i32;
        w[j] = p[b];
        sum += p[b];
    }
    // As llama.cpp does: never divide by less than the smallest normal FP16.
    const MIN_DENOM: f32 = 1.0 / 16384.0;
    let denom = if sum < MIN_DENOM { MIN_DENOM } else { sum };
    for v in w {
        *v /= denom;
    }
}

/// RMSNorm without gamma: y = x / sqrt(mean(x^2) + eps).
pub(crate) fn rmsnorm(y: &mut [f32], x: &[f32], eps: f32) {
    Backend::current().rmsnorm(y, x, eps)
}

pub(crate) fn rmsnorm_inplace(x: &mut [f32], eps: f32) {
    Backend::current().rmsnorm_inplace(x, eps)
}

/// RMSNorm times `gamma[..dim]` over each of the `x.len() / dim` groups of
/// `x` (into the first x.len() floats of `y`), gamma staying in L1.
pub(crate) fn rmsnorm_gamma_multi(y: &mut [f32], x: &[f32], dim: usize, gamma: &[f32], eps: f32) {
    Backend::current().rmsnorm_gamma_multi(y, x, dim, gamma, eps)
}

pub(crate) fn rmsnorm_gamma_multi_inplace(x: &mut [f32], dim: usize, gamma: &[f32], eps: f32) {
    Backend::current().rmsnorm_gamma_multi_inplace(x, dim, gamma, eps)
}

/// Scale each group of `dim` floats to length `scale`:
/// x *= scale / sqrt(sum(x^2) + eps).
pub(crate) fn l2norm_multi(x: &mut [f32], dim: usize, eps: f32, scale: f32) {
    Backend::current().l2norm_multi(x, dim, eps, scale)
}

/// sum a[i] * b[i]; n = a.len(). Four accumulators: for attention-sized
/// and router dots.
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    Backend::current().dot(a, b)
}

/// dst = bf16 -> f32 of src; n = dst.len().
pub(crate) fn bf16_to_f32_row(src: &[u16], dst: &mut [f32]) {
    Backend::current().bf16_to_f32_row(src, dst)
}

/// Rotary position embedding of one token at `pos`: the first `n_rot`
/// lanes of each of `n_heads` heads (`head_dim` apart) rotate in pairs by
/// theta_i = pos * freq_base^(-2i/n_rot) / freq_factors[i]; the rest pass
/// through. `freq_factors` (n_rot/2 floats) carries Llama 3's long-context
/// scaling and Gemma 4's proportional RoPE (1e30 switches a pair off).
/// Linear position scaling is the caller's: pass pos * scale.
pub(crate) fn rope(
    x: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    n_rot: usize,
    pos: f32,
    freq_base: f32,
    mode: RopeMode,
    freq_factors: Option<&[f32]>,
) {
    if n_rot == 0 {
        return;
    }
    const MAX_HALF: usize = 256; // head_dim <= 512 for every model we load
    let half = n_rot / 2;
    assert!(
        n_rot.is_multiple_of(2) && n_rot <= head_dim && half <= MAX_HALF,
        "rope: n_rot {n_rot}, head_dim {head_dim}"
    );
    let x = &mut x[..elems(n_heads, head_dim)];
    let freq_factors = freq_factors.map(|f| &f[..half]);
    // theta_i depends only on the lane, so the cos/sin pairs serve every
    // head. freq_base^(-2/n_rot) is the ratio of the geometric schedule:
    // compute it once and accumulate (one powf rather than one per pair).
    let (mut cs, mut sn) = ([0f32; MAX_HALF], [0f32; MAX_HALF]);
    let ratio = freq_base.powf(-2.0 / n_rot as f32);
    let mut inv = 1.0f32;
    for i in 0..half {
        let mut theta = pos * inv;
        if let Some(ff) = freq_factors {
            theta /= ff[i];
        }
        cs[i] = theta.cos();
        sn[i] = theta.sin();
        inv *= ratio;
    }
    // Which pairing a model needs is fixed by how its Q/K weights were laid
    // out at conversion time.
    let (stride, partner) = match mode {
        RopeMode::Neox => (1, half),
        RopeMode::Norm => (2, 1),
    };
    for xh in x.chunks_exact_mut(head_dim) {
        for i in 0..half {
            let p = i * stride;
            let (a, b) = (xh[p], xh[p + partner]);
            xh[p] = cs[i] * a - sn[i] * b;
            xh[p + partner] = sn[i] * a + cs[i] * b;
        }
    }
}

/// Causal depthwise convolution over time, then SiLU. `x` holds k - 1 +
/// n_tok rows of `ch` floats (the k - 1 inputs before the batch, then the
/// batch's own) and `w` the taps as k rows of `ch`, oldest first:
///   y[t][c] = silu(sum_j w[j][c] * x[t + j][c])   for t in [0, n_tok).
pub(crate) fn conv1d_silu(y: &mut [f32], x: &[f32], w: &[f32], n_tok: usize, ch: usize, k: usize) {
    Backend::current().conv1d_silu(y, x, w, n_tok, ch, k)
}

/// One token of the gated delta rule for one head. The state `s` is d_v
/// rows of d_k floats (row j: what value lane j has stored along each key
/// lane). With k unit-length and q unit-length times 1/sqrt(d_k):
///   S = decay S;  S += beta (v - S k) k^T;  o = S q.
pub(crate) fn delta_rule(
    s: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    decay: f32,
    beta: f32,
    o: &mut [f32],
    d_k: usize,
    d_v: usize,
) {
    Backend::current().delta_rule(s, q, k, v, decay, beta, o, d_k, d_v)
}

/// One block of `n_keys` keys for `n_heads` query heads sharing one KV head
/// (each K/V row is read once for all of them): the heads' queries are
/// `q_stride` floats apart in `q`. `scores` is scratch for n_heads * n_keys
/// floats; `part` receives n_heads partials of [`attn_part`] floats.
/// Merge partials in block order with [`attn_merge`], then [`attn_finish`].
/// The arithmetic depends only on where the blocks fall, so fixed blocks
/// give the same bits whether they run in sequence or on different threads.
pub(crate) fn attn_block(
    q: &[f32],
    q_stride: usize,
    k: KvRows<'_>,
    v: KvRows<'_>,
    kv_stride: usize,
    n_keys: usize,
    head_dim: usize,
    n_heads: usize,
    scale: f32,
    scores: &mut [f32],
    part: &mut [f32],
) {
    Backend::current().attn_block(
        q, q_stride, k, v, kv_stride, n_keys, head_dim, n_heads, scale, scores, part,
    )
}

/// acc <- the merge of acc (earlier keys) and part (the next block).
pub(crate) fn attn_merge(acc: &mut [f32], part: &[f32], head_dim: usize, n_heads: usize) {
    Backend::current().attn_merge(acc, part, head_dim, n_heads)
}

/// out[h] = o / l for each head, `out_stride` floats apart.
pub(crate) fn attn_finish(out: &mut [f32], out_stride: usize, acc: &[f32], head_dim: usize, n_heads: usize) {
    let acc = &acc[..elems(n_heads, attn_part(head_dim))];
    let out = &mut out[..strided_len(n_heads, out_stride, head_dim)];
    scalar::attn_finish(out, out_stride, acc, head_dim, n_heads)
}

/// BF16 weights `w` [rows x cols], f32 x and y: y[..rows] = W x.
pub(crate) fn gemv_bf16(w: &[u16], x: &[f32], y: &mut [f32], rows: usize, cols: usize) {
    Backend::current().gemv_bf16(w, x, y, rows, cols)
}

/// The index of the largest element of the BF16 gemv, without writing it
/// (greedy lm_head; soft-capping is monotonic, so skip it). rows > 0.
pub(crate) fn gemv_bf16_argmax(w: &[u16], x: &[f32], rows: usize, cols: usize) -> usize {
    Backend::current().gemv_bf16_argmax(w, x, rows, cols)
}

/// Q8_0 split layout: y[r] = sum_b scales[r, b] * sum_{k in b} qs[r, k] x[k].
/// cols must be a multiple of 32.
pub(crate) fn gemv_q8_0(qs: &[i8], scales: &[u16], x: &[f32], y: &mut [f32], rows: usize, cols: usize) {
    Backend::current().gemv_q8_0(qs, scales, x, y, rows, cols)
}

/// Q4 split layout, w = scale * q + bias: y[r] = sum_b scales[r, b] *
/// sum_{k in b} q[r, k] x[k] + dot(biases[r, ..], xsum), where `xsum` holds
/// x's cols/32 block sums ([`block_sums`]). cols must be a multiple of 32.
pub(crate) fn gemv_q4(
    qs: &[u8],
    scales: &[u16],
    biases: &[u16],
    x: &[f32],
    xsum: &[f32],
    y: &mut [f32],
    rows: usize,
    cols: usize,
) {
    Backend::current().gemv_q4(qs, scales, biases, x, xsum, y, rows, cols)
}

/// out[b] = sum of x[32b .. 32b + 32] for b < x.len() / 32 (x.len() a
/// multiple of 32).
pub(crate) fn block_sums(x: &[f32], out: &mut [f32]) {
    Backend::current().block_sums(x, out)
}

/// [`gemv_bf16`] for `n_batch` vectors of x (`x_stride` apart) into
/// `n_batch` vectors of y (`y_stride` >= rows apart), one pass over W.
pub(crate) fn gemv_bf16_batch(
    w: &[u16],
    x: &[f32],
    x_stride: usize,
    y: &mut [f32],
    y_stride: usize,
    rows: usize,
    cols: usize,
    n_batch: usize,
) {
    Backend::current().gemv_bf16_batch(w, x, x_stride, y, y_stride, rows, cols, n_batch)
}

/// [`gemv_q8_0`] for a batch, as [`gemv_bf16_batch`]; each vector gets the
/// bits the single-vector gemv gives it.
pub(crate) fn gemv_q8_0_batch(
    qs: &[i8],
    scales: &[u16],
    x: &[f32],
    x_stride: usize,
    y: &mut [f32],
    y_stride: usize,
    rows: usize,
    cols: usize,
    n_batch: usize,
) {
    Backend::current().gemv_q8_0_batch(qs, scales, x, x_stride, y, y_stride, rows, cols, n_batch)
}

/// [`gemv_q4`] for a batch, as [`gemv_q8_0_batch`]; vector n's block sums
/// are at `xsum[n * xsum_stride..]`.
pub(crate) fn gemv_q4_batch(
    qs: &[u8],
    scales: &[u16],
    biases: &[u16],
    x: &[f32],
    x_stride: usize,
    xsum: &[f32],
    xsum_stride: usize,
    y: &mut [f32],
    y_stride: usize,
    rows: usize,
    cols: usize,
    n_batch: usize,
) {
    Backend::current().gemv_q4_batch(
        qs,
        scales,
        biases,
        x,
        x_stride,
        xsum,
        xsum_stride,
        y,
        y_stride,
        rows,
        cols,
        n_batch,
    )
}

/// y = row `token` of a BF16 table, times `scale`; n_embd = y.len().
pub(crate) fn embed_row_bf16(y: &mut [f32], table: &[u16], token: usize, scale: f32) {
    let n = y.len();
    bf16_to_f32_row(&table[elems(token, n)..][..n], y);
    if scale != 1.0 {
        for v in y {
            *v *= scale;
        }
    }
}

/// y = row `token` of a Q8_0 split-layout table, times `scale`; y.len() a
/// multiple of 32.
pub(crate) fn embed_row_q8_0(y: &mut [f32], qs: &[i8], scales: &[u16], token: usize, scale: f32) {
    let n = y.len();
    assert!(
        n.is_multiple_of(32),
        "embed_row_q8_0: n_embd {n} is not a multiple of 32"
    );
    let (q, s) = (&qs[elems(token, n)..][..n], &scales[elems(token, n / 32)..][..n / 32]);
    for ((y, q), &s) in y.as_chunks_mut::<32>().0.iter_mut().zip(q.as_chunks::<32>().0).zip(s) {
        let vs = bf16_to_f32(s) * scale;
        for (y, &q) in y.iter_mut().zip(q) {
            *y = f32::from(q) * vs;
        }
    }
}

/// y = row `token` of a Q4 split-layout table, times `scale`; y.len() a
/// multiple of 32.
pub(crate) fn embed_row_q4(y: &mut [f32], qs: &[u8], scales: &[u16], biases: &[u16], token: usize, scale: f32) {
    let n = y.len();
    assert!(n.is_multiple_of(32), "embed_row_q4: n_embd {n} is not a multiple of 32");
    let nb = n / 32;
    let q = &qs[elems(token, n / 2)..][..n / 2];
    let (s, b) = (&scales[elems(token, nb)..][..nb], &biases[elems(token, nb)..][..nb]);
    for (((y, q), &s), &b) in y
        .as_chunks_mut::<32>()
        .0
        .iter_mut()
        .zip(q.as_chunks::<16>().0)
        .zip(s)
        .zip(b)
    {
        let (s, bias) = (bf16_to_f32(s) * scale, bf16_to_f32(b) * scale);
        for k in 0..16 {
            y[k] = s * f32::from(q[k] & 0x0f) + bias;
            y[k + 16] = s * f32::from(q[k] >> 4) + bias;
        }
    }
}
