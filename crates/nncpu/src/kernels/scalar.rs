//! Portable kernels.
//!
//! The AVX2 kernels' arithmetic, lane by lane: an 8-wide accumulator is an
//! `[f32; 8]` reduced in the same order, and tails are handled the same way.
//! Where AVX2 fuses a multiply-add these use [`fmadd`], which fuses too when
//! the target has FMA at compile time (the results then match the AVX2
//! kernels bit for bit) and rounds twice otherwise. The `*_lane` functions
//! take the multiply-add as a parameter so that the AVX2 kernels can run
//! their tails through them with a fused one.

use super::KvRows;
use crate::pool::RowSpan;
use crate::runtime::bf16_to_f32;
use std::array::from_fn;
use std::ops::Range;

type V8 = [f32; 8];

// exp(x): x = n*ln(2) + r, a degree-5 polynomial for exp(r), then 2^n built
// in the exponent field. The literals are llmcpu's.
#[allow(clippy::excessive_precision)]
pub(super) const EXP_LO: f32 = -87.336544; // ~ln(FLT_MIN)
#[allow(clippy::excessive_precision)]
pub(super) const EXP_HI: f32 = 88.722839; // ~ln(FLT_MAX)
pub(super) const LOG2E: f32 = std::f32::consts::LOG2_E;
#[allow(clippy::excessive_precision)]
pub(super) const LN2_HI: f32 = 0.693359375;
#[allow(clippy::excessive_precision)]
pub(super) const LN2_LO: f32 = -2.12194440e-4;
#[allow(clippy::excessive_precision)]
pub(super) const EXP_P: [f32; 6] = [
    1.9875691500e-4,
    1.3981999507e-3,
    8.3334519073e-3,
    4.1665795894e-2,
    1.6666665459e-1,
    5.0000001201e-1,
];

// GELU(x) = 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))), with tanh a
// Pade [7/6] approximant that saturates to sign(u) for |u| >= 5.
#[allow(clippy::excessive_precision)]
pub(super) const GELU_C: f32 = 0.79788456080286535; // sqrt(2/pi)
pub(super) const GELU_A: f32 = 0.044715;

/// `a * b + c`, fused where the target has FMA at compile time.
#[inline(always)]
pub(super) fn fmadd(a: f32, b: f32, c: f32) -> f32 {
    if cfg!(any(target_feature = "fma", target_arch = "aarch64")) {
        a.mul_add(b, c)
    } else {
        a * b + c
    }
}

/// Round to nearest, ties to even, for |v| < 2^22: adding 1.5 * 2^23 leaves
/// no fraction bits, so the addition itself rounds (what `vroundps` does).
#[inline(always)]
fn round_even(v: f32) -> f32 {
    const MAGIC: f32 = 12_582_912.0;
    (v + MAGIC) - MAGIC
}

/// One lane of the AVX2 `exp`. Max relative error ~3e-7 over
/// [-87.3, 88.7]; outside, it saturates (to ~1.2e-38 below, which softmax
/// masks out for -inf inputs).
#[inline(always)]
pub(super) fn exp_lane(x: f32, fma: impl Fn(f32, f32, f32) -> f32) -> f32 {
    // max/min as maxps/minps order them: a NaN takes the bound.
    let x = if x > EXP_LO { x } else { EXP_LO };
    let x = if x < EXP_HI { x } else { EXP_HI };
    let fx = round_even(x * LOG2E);
    // r = x - fx*ln(2), in two halves for accuracy.
    let r = fma(-fx, LN2_HI, x);
    let r = fma(-fx, LN2_LO, r);
    let mut y = fma(EXP_P[0], r, EXP_P[1]);
    y = fma(y, r, EXP_P[2]);
    y = fma(y, r, EXP_P[3]);
    y = fma(y, r, EXP_P[4]);
    y = fma(y, r, EXP_P[5]);
    y = fma(y, r * r, r);
    y += 1.0;
    y * f32::from_bits(((fx as i32 + 127) as u32) << 23)
}

/// One lane of the AVX2 GELU (tanh form, Pade tanh).
#[inline(always)]
pub(super) fn gelu_lane(v: f32, fma: impl Fn(f32, f32, f32) -> f32) -> f32 {
    let v3 = (v * v) * v;
    let u = GELU_C * fma(GELU_A, v3, v);
    let u2 = u * u;
    let u4 = u2 * u2;
    let u6 = u4 * u2;
    // tanh(u) ~= u (135135 + 17325 u^2 + 378 u^4 + u^6)
    //           / (135135 + 62370 u^2 + 3150 u^4 + 28 u^6)
    let num = u * (u6 + fma(378.0, u4, fma(17325.0, u2, 135135.0)));
    let den = fma(28.0, u6, fma(3150.0, u4, fma(62370.0, u2, 135135.0)));
    let t = if u.abs() >= 5.0 { 1f32.copysign(u) } else { num / den };
    (0.5 * v) * (1.0 + t)
}

/// One lane of the AVX2 sigmoid: 1 / (1 + exp(0 - g)), a true divide.
#[inline(always)]
pub(super) fn sigmoid_lane(g: f32, fma: impl Fn(f32, f32, f32) -> f32) -> f32 {
    1.0 / (1.0 + exp_lane(0.0 - g, fma))
}

#[inline(always)]
fn add8(a: V8, b: V8) -> V8 {
    from_fn(|l| a[l] + b[l])
}

/// The AVX2 horizontal sum: high half onto low, then two `hadd`s.
#[inline(always)]
fn hsum8(v: V8) -> f32 {
    ((v[0] + v[4]) + (v[1] + v[5])) + ((v[2] + v[6]) + (v[3] + v[7]))
}

/// `maxps`: the second operand unless the first is greater.
#[inline(always)]
fn maxps(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

/// The AVX2 horizontal max.
#[inline(always)]
fn hmax8(v: V8) -> f32 {
    let m: [f32; 4] = from_fn(|l| maxps(v[l], v[l + 4]));
    maxps(maxps(m[0], m[2]), maxps(m[1], m[3]))
}

// ---- elementwise ------------------------------------------------------------

pub(super) fn gelu_inplace(x: &mut [f32]) {
    for v in x {
        *v = gelu_lane(*v, fmadd);
    }
}

pub(super) fn gelu_mul_inplace(gate: &mut [f32], up: &[f32]) {
    for (g, &u) in gate.iter_mut().zip(up) {
        *g = gelu_lane(*g, fmadd) * u;
    }
}

pub(super) fn swiglu_inplace(gate: &mut [f32], up: &[f32]) {
    for (g, &u) in gate.iter_mut().zip(up) {
        *g = (*g * sigmoid_lane(*g, fmadd)) * u;
    }
}

pub(super) fn silu_mul_into(up: &mut [f32], gate: &[f32]) {
    for (u, &g) in up.iter_mut().zip(gate) {
        *u *= g * sigmoid_lane(g, fmadd);
    }
}

pub(super) fn mul_sigmoid(y: &mut [f32], g: &[f32]) {
    for (y, &g) in y.iter_mut().zip(g) {
        *y *= sigmoid_lane(g, fmadd);
    }
}

pub(super) fn add_inplace(y: &mut [f32], b: &[f32]) {
    for (y, &b) in y.iter_mut().zip(b) {
        *y += b;
    }
}

pub(super) fn mul_inplace(y: &mut [f32], s: &[f32]) {
    for (y, &s) in y.iter_mut().zip(s) {
        *y *= s;
    }
}

pub(super) fn softmax_inplace(p: &mut [f32]) {
    let n = p.len();
    if n == 0 {
        return;
    }
    let body = n & !7;
    let mut m = f32::NEG_INFINITY;
    if n >= 8 {
        let mut vmax: V8 = from_fn(|l| p[l]);
        for c in p[8..body].as_chunks::<8>().0 {
            for l in 0..8 {
                vmax[l] = maxps(vmax[l], c[l]);
            }
        }
        m = hmax8(vmax);
    }
    for &v in &p[body..] {
        if v > m {
            m = v;
        }
    }
    // exp(p - m); -inf entries (masked keys) become exactly 0.
    let mut vsum = [0f32; 8];
    for c in p[..body].as_chunks_mut::<8>().0 {
        for l in 0..8 {
            let v = if c[l] == f32::NEG_INFINITY {
                0.0
            } else {
                exp_lane(c[l] - m, fmadd)
            };
            c[l] = v;
            vsum[l] += v;
        }
    }
    let mut sum = f64::from(hsum8(vsum));
    for v in &mut p[body..] {
        if *v == f32::NEG_INFINITY {
            *v = 0.0;
        } else {
            *v = exp_lane(*v - m, fmadd);
            sum += f64::from(*v);
        }
    }
    // A fully masked row is already all zeros: no NaN.
    let inv = if sum > 0.0 { (1.0 / sum) as f32 } else { 0.0 };
    for v in p {
        *v *= inv;
    }
}

// ---- norms -----------------------------------------------------------------

/// Sum of squares as the AVX2 norms take it: FMA lanes over whole 8-chunks
/// (only when there is at least one), then the tail in double.
fn sum_sq(x: &[f32]) -> f64 {
    let body = x.len() & !7;
    let mut s = 0.0;
    if x.len() >= 8 {
        let mut acc = [0f32; 8];
        for c in x[..body].as_chunks::<8>().0 {
            for l in 0..8 {
                acc[l] = fmadd(c[l], c[l], acc[l]);
            }
        }
        s = f64::from(hsum8(acc));
    }
    for &v in &x[body..] {
        s += f64::from(v) * f64::from(v);
    }
    s
}

pub(super) fn rmsnorm(y: &mut [f32], x: &[f32], eps: f32) {
    if x.is_empty() {
        return;
    }
    let scale = (1.0 / (sum_sq(x) / x.len() as f64 + f64::from(eps)).sqrt()) as f32;
    for (y, &x) in y.iter_mut().zip(x) {
        *y = x * scale;
    }
}

pub(super) fn rmsnorm_inplace(x: &mut [f32], eps: f32) {
    if x.is_empty() {
        return;
    }
    let scale = (1.0 / (sum_sq(x) / x.len() as f64 + f64::from(eps)).sqrt()) as f32;
    for v in x {
        *v *= scale;
    }
}

/// The per-group scale of [`rmsnorm_gamma_multi`]: note `* (1/dim)` in f32,
/// not `/ dim`.
fn gamma_scale(xg: &[f32], inv_dim: f32, eps: f32) -> f32 {
    (1.0 / (sum_sq(xg) * f64::from(inv_dim) + f64::from(eps)).sqrt()) as f32
}

pub(super) fn rmsnorm_gamma_multi(y: &mut [f32], x: &[f32], dim: usize, gamma: &[f32], eps: f32) {
    let inv_dim = 1.0 / dim as f32;
    for (yg, xg) in y.chunks_exact_mut(dim).zip(x.chunks_exact(dim)) {
        let scale = gamma_scale(xg, inv_dim, eps);
        for ((y, &x), &g) in yg.iter_mut().zip(xg).zip(gamma) {
            *y = x * scale * g;
        }
    }
}

pub(super) fn rmsnorm_gamma_multi_inplace(x: &mut [f32], dim: usize, gamma: &[f32], eps: f32) {
    let inv_dim = 1.0 / dim as f32;
    for xg in x.chunks_exact_mut(dim) {
        let scale = gamma_scale(xg, inv_dim, eps);
        for (v, &g) in xg.iter_mut().zip(gamma) {
            *v = *v * scale * g;
        }
    }
}

pub(super) fn l2norm_multi(x: &mut [f32], dim: usize, eps: f32, scale: f32) {
    for xg in x.chunks_exact_mut(dim) {
        let s = scale / (dot(xg, xg) + eps).sqrt();
        for v in xg {
            *v *= s;
        }
    }
}

// ---- dots and conversions -----------------------------------------------------

/// Four 8-lane accumulators over 32-element strides, the leftover 8-chunks
/// into the first, a `(s0 + s1) + (s2 + s3)` reduction, then a scalar tail.
pub(super) fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let mut s = [[0f32; 8]; 4];
    let mut i = 0;
    while i + 32 <= n {
        for (j, acc) in s.iter_mut().enumerate() {
            let (a, b) = (&a[i + 8 * j..][..8], &b[i + 8 * j..][..8]);
            for l in 0..8 {
                acc[l] = fmadd(a[l], b[l], acc[l]);
            }
        }
        i += 32;
    }
    while i + 8 <= n {
        let (a, b) = (&a[i..i + 8], &b[i..i + 8]);
        for l in 0..8 {
            s[0][l] = fmadd(a[l], b[l], s[0][l]);
        }
        i += 8;
    }
    let mut acc = hsum8(add8(add8(s[0], s[1]), add8(s[2], s[3])));
    for (&a, &b) in a[i..].iter().zip(&b[i..]) {
        acc += a * b;
    }
    acc
}

pub(super) fn dot_bf16_f32(a: &[u16], b: &[f32]) -> f32 {
    let body = a.len() & !7;
    let mut acc = [0f32; 8];
    for (a, b) in a[..body].as_chunks::<8>().0.iter().zip(b[..body].as_chunks::<8>().0) {
        for l in 0..8 {
            acc[l] = fmadd(bf16_to_f32(a[l]), b[l], acc[l]);
        }
    }
    let mut s = hsum8(acc);
    for (&a, &b) in a[body..].iter().zip(&b[body..]) {
        s += bf16_to_f32(a) * b;
    }
    s
}

pub(super) fn bf16_to_f32_row(src: &[u16], dst: &mut [f32]) {
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = bf16_to_f32(s);
    }
}

pub(super) fn block_sums(x: &[f32], out: &mut [f32]) {
    for (c, o) in x.as_chunks::<32>().0.iter().zip(out) {
        *o = hsum8(from_fn(|l| ((c[l] + c[8 + l]) + c[16 + l]) + c[24 + l]));
    }
}

// ---- Gated DeltaNet -----------------------------------------------------------

pub(super) fn conv1d_silu(y: &mut [f32], x: &[f32], w: &[f32], n_tok: usize, ch: usize, k: usize) {
    for (t, yt) in y.chunks_exact_mut(ch).take(n_tok).enumerate() {
        let xt = &x[t * ch..];
        for (c, y) in yt.iter_mut().enumerate() {
            let mut acc = 0.0;
            for j in 0..k {
                acc = fmadd(w[j * ch + c], xt[j * ch + c], acc);
            }
            *y = acc / (1.0 + exp_lane(0.0 - acc, fmadd));
        }
    }
}

pub(super) fn delta_rule(
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
    for (j, row) in s.chunks_exact_mut(d_k).take(d_v).enumerate() {
        // The decayed row's projection on k, then the step towards v[j].
        let delta = beta * (v[j] - decay * dot(row, k));
        let (mut acc0, mut acc1) = ([0f32; 8], [0f32; 8]);
        let mut i = 0;
        while i + 16 <= d_k {
            for l in 0..8 {
                let s0 = fmadd(delta, k[i + l], row[i + l] * decay);
                let s1 = fmadd(delta, k[i + 8 + l], row[i + 8 + l] * decay);
                row[i + l] = s0;
                row[i + 8 + l] = s1;
                acc0[l] = fmadd(s0, q[i + l], acc0[l]);
                acc1[l] = fmadd(s1, q[i + 8 + l], acc1[l]);
            }
            i += 16;
        }
        while i + 8 <= d_k {
            for l in 0..8 {
                let s0 = fmadd(delta, k[i + l], row[i + l] * decay);
                row[i + l] = s0;
                acc0[l] = fmadd(s0, q[i + l], acc0[l]);
            }
            i += 8;
        }
        let mut out = hsum8(add8(acc0, acc1));
        for i in i..d_k {
            let s = row[i] * decay + delta * k[i];
            row[i] = s;
            out += s * q[i];
        }
        o[j] = out;
    }
}

// ---- attention ------------------------------------------------------------------

pub(super) fn attn_block(
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
    let shape = (q_stride, kv_stride, n_keys, head_dim, n_heads, scale);
    match (k, v) {
        (KvRows::F32(k), KvRows::F32(v)) => attn_block_t(q, k, v, shape, scores, part, |x| x),
        (KvRows::Bf16(k), KvRows::Bf16(v)) => attn_block_t(q, k, v, shape, scores, part, bf16_to_f32),
        _ => unreachable!("validated by the caller"),
    }
}

fn attn_block_t<T: Copy>(
    q: &[f32],
    kr: &[T],
    vr: &[T],
    (q_stride, kv_stride, n_keys, hd, n_heads, scale): (usize, usize, usize, usize, usize, f32),
    scores: &mut [f32],
    part: &mut [f32],
    f: impl Fn(T) -> f32,
) {
    let pw = hd + 2;
    let hd8 = hd & !7;
    // Scores (each head's accumulator is its own, so grouping heads changes
    // nothing here).
    for t in 0..n_keys {
        let krow = &kr[t * kv_stride..][..hd];
        for h in 0..n_heads {
            let qh = &q[h * q_stride..][..hd];
            let mut acc = [0f32; 8];
            for i in (0..hd8).step_by(8) {
                for l in 0..8 {
                    acc[l] = fmadd(qh[i + l], f(krow[i + l]), acc[l]);
                }
            }
            let mut s = hsum8(acc);
            for i in hd8..hd {
                s += qh[i] * f(krow[i]);
            }
            scores[h * n_keys + t] = s * scale;
        }
    }
    // Each head's weights exp(s - m), with m the block's own max.
    let body = n_keys & !7;
    for h in 0..n_heads {
        let s = &mut scores[h * n_keys..][..n_keys];
        let mut m = s[0];
        for &v in &s[1..] {
            if m < v {
                m = v;
            }
        }
        let mut vsum = [0f32; 8];
        for c in s[..body].as_chunks_mut::<8>().0 {
            for l in 0..8 {
                c[l] = exp_lane(c[l] - m, fmadd);
                vsum[l] += c[l];
            }
        }
        let mut l = hsum8(vsum);
        for w in &mut s[body..] {
            *w = exp_lane(*w - m, fmadd);
            l += *w;
        }
        part[h * pw + hd] = m;
        part[h * pw + hd + 1] = l;
    }
    // Weighted sum of V rows: the first row writes, the rest accumulate.
    for t in 0..n_keys {
        let vrow = &vr[t * kv_stride..][..hd];
        for h in 0..n_heads {
            let w = scores[h * n_keys + t];
            let o = &mut part[h * pw..][..hd];
            if t == 0 {
                for (o, &v) in o.iter_mut().zip(vrow) {
                    *o = w * f(v);
                }
            } else {
                for (o, &v) in o[..hd8].iter_mut().zip(vrow) {
                    *o = fmadd(w, f(v), *o);
                }
                for (o, &v) in o[hd8..].iter_mut().zip(&vrow[hd8..]) {
                    *o += w * f(v);
                }
            }
        }
    }
}

pub(super) fn attn_merge(acc: &mut [f32], part: &[f32], head_dim: usize, n_heads: usize) {
    let (hd, pw) = (head_dim, head_dim + 2);
    let hd8 = hd & !7;
    for (a, b) in acc.chunks_exact_mut(pw).zip(part.chunks_exact(pw)).take(n_heads) {
        let m = if a[hd] < b[hd] { b[hd] } else { a[hd] };
        let ca = (a[hd] - m).exp();
        let cb = (b[hd] - m).exp();
        for i in 0..hd8 {
            a[i] = fmadd(b[i], cb, a[i] * ca);
        }
        for i in hd8..hd {
            a[i] = a[i] * ca + b[i] * cb;
        }
        a[hd] = m;
        a[hd + 1] = a[hd + 1] * ca + b[hd + 1] * cb;
    }
}

pub(super) fn attn_finish(out: &mut [f32], out_stride: usize, acc: &[f32], head_dim: usize, n_heads: usize) {
    let pw = head_dim + 2;
    for (h, a) in acc.chunks_exact(pw).take(n_heads).enumerate() {
        let l = a[head_dim + 1];
        let inv = if l > 0.0 { 1.0 / l } else { 0.0 };
        for (o, &a) in out[h * out_stride..][..head_dim].iter_mut().zip(a) {
            *o = a * inv;
        }
    }
}

// ---- gemv ---------------------------------------------------------------------------

/// The 4-row BF16 kernel: one 8-lane accumulator per row, each x tile shared
/// by the four rows, and the `hadd` tree's reduction order.
#[inline(always)]
fn bf16_4rows(w: &[u16], cols: usize, x: &[f32]) -> [f32; 4] {
    let mut acc = [[0f32; 8]; 4];
    for k in (0..cols).step_by(8) {
        let xt = &x[k..k + 8];
        for (j, a) in acc.iter_mut().enumerate() {
            let wt = &w[j * cols + k..][..8];
            for l in 0..8 {
                a[l] = fmadd(bf16_to_f32(wt[l]), xt[l], a[l]);
            }
        }
    }
    acc.map(|a| ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7])))
}

/// The 1-row BF16 kernel: four accumulators when cols % 32 == 0, else one.
fn bf16_row(w: &[u16], x: &[f32], cols: usize) -> f32 {
    if cols.is_multiple_of(32) {
        let mut acc = [[0f32; 8]; 4];
        for k in (0..cols).step_by(32) {
            for (j, a) in acc.iter_mut().enumerate() {
                let (wt, xt) = (&w[k + 8 * j..][..8], &x[k + 8 * j..][..8]);
                for l in 0..8 {
                    a[l] = fmadd(bf16_to_f32(wt[l]), xt[l], a[l]);
                }
            }
        }
        hsum8(add8(add8(acc[0], acc[1]), add8(acc[2], acc[3])))
    } else {
        let mut a = [0f32; 8];
        for (wt, xt) in w[..cols].as_chunks::<8>().0.iter().zip(x.as_chunks::<8>().0) {
            for l in 0..8 {
                a[l] = fmadd(bf16_to_f32(wt[l]), xt[l], a[l]);
            }
        }
        hsum8(a)
    }
}

pub(super) fn gemv_bf16_quads(w: &[u16], x: &[f32], x_stride: usize, cols: usize, out: &mut RowSpan<'_, f32>) {
    let first = out.first();
    for i in (0..out.len()).step_by(4) {
        let r = first + i;
        let wq = &w[r * cols..(r + 4) * cols];
        for n in 0..out.n_vec() {
            let y = bf16_4rows(wq, cols, &x[n * x_stride..][..cols]);
            out.vec(n)[i..i + 4].copy_from_slice(&y);
        }
    }
}

pub(super) fn gemv_bf16_rows(w: &[u16], x: &[f32], cols: usize, out: &mut RowSpan<'_, f32>) {
    let first = out.first();
    for (i, y) in out.vec(0).iter_mut().enumerate() {
        *y = bf16_row(&w[(first + i) * cols..][..cols], x, cols);
    }
}

pub(super) fn gemv_bf16_argmax_quads(w: &[u16], x: &[f32], cols: usize, quads: Range<usize>) -> Option<(f32, usize)> {
    let mut best = None;
    let mut bv = f32::NEG_INFINITY;
    for q in quads {
        let r = 4 * q;
        for (j, v) in bf16_4rows(&w[r * cols..(r + 4) * cols], cols, x)
            .into_iter()
            .enumerate()
        {
            if v > bv {
                bv = v;
                best = Some((v, r + j));
            }
        }
    }
    best
}

/// The Q8_0 row kernel for N tokens: each block's weights are widened and
/// scaled once and fed to every token's four accumulators, so each token's
/// sum has the bits the 1-token kernel gives.
#[inline(always)]
fn q8_row<const N: usize>(q: &[i8], s: &[u16], xs: [&[f32]; N]) -> [f32; N] {
    let mut acc = [[[0f32; 8]; 4]; N];
    for (b, (qb, &sb)) in q.as_chunks::<32>().0.iter().zip(s).enumerate() {
        let sc = bf16_to_f32(sb);
        for k in 0..4 {
            let w: V8 = from_fn(|l| f32::from(qb[8 * k + l]) * sc);
            for (a, x) in acc.iter_mut().zip(&xs) {
                let xt = &x[32 * b + 8 * k..][..8];
                for l in 0..8 {
                    a[k][l] = fmadd(w[l], xt[l], a[k][l]);
                }
            }
        }
    }
    acc.map(|a| hsum8(add8(add8(a[0], a[1]), add8(a[2], a[3]))))
}

/// The Q4 row kernel for N tokens (the bias term is the caller's). Byte i
/// of a block holds element i in its low nibble and i + 16 in its high one.
#[inline(always)]
fn q4_row<const N: usize>(q: &[u8], s: &[u16], xs: [&[f32]; N]) -> [f32; N] {
    let mut acc = [[[0f32; 8]; 4]; N];
    for (b, (qb, &sb)) in q.as_chunks::<16>().0.iter().zip(s).enumerate() {
        let sc = bf16_to_f32(sb);
        let w: [V8; 4] = [
            from_fn(|l| f32::from(qb[l] & 15) * sc),
            from_fn(|l| f32::from(qb[8 + l] & 15) * sc),
            from_fn(|l| f32::from(qb[l] >> 4) * sc),
            from_fn(|l| f32::from(qb[8 + l] >> 4) * sc),
        ];
        for (a, x) in acc.iter_mut().zip(&xs) {
            for (k, w) in w.iter().enumerate() {
                let xt = &x[32 * b + 8 * k..][..8];
                for l in 0..8 {
                    a[k][l] = fmadd(w[l], xt[l], a[k][l]);
                }
            }
        }
    }
    acc.map(|a| hsum8(add8(add8(a[0], a[1]), add8(a[2], a[3]))))
}

/// Rows of a Q8_0 gemv for every vector of `out`: tokens three at a time,
/// the rest one by one.
pub(super) fn gemv_q8_0_rows(
    qs: &[i8],
    scales: &[u16],
    x: &[f32],
    x_stride: usize,
    cols: usize,
    out: &mut RowSpan<'_, f32>,
) {
    let nb = cols / 32;
    let first = out.first();
    let xv = |n: usize| &x[n * x_stride..][..cols];
    for i in 0..out.len() {
        let r = first + i;
        let (q, s) = (&qs[r * cols..][..cols], &scales[r * nb..][..nb]);
        let mut n = 0;
        while n + 3 <= out.n_vec() {
            for (t, y) in q8_row(q, s, [xv(n), xv(n + 1), xv(n + 2)]).into_iter().enumerate() {
                out.vec(n + t)[i] = y;
            }
            n += 3;
        }
        for n in n..out.n_vec() {
            let [y] = q8_row(q, s, [xv(n)]);
            out.vec(n)[i] = y;
        }
    }
}

/// Rows of a Q4 gemv for every vector of `out`, as [`gemv_q8_0_rows`], each
/// row's sum plus `dot(biases, xsum)`.
pub(super) fn gemv_q4_rows(
    qs: &[u8],
    scales: &[u16],
    biases: &[u16],
    x: &[f32],
    x_stride: usize,
    xsum: &[f32],
    xsum_stride: usize,
    cols: usize,
    out: &mut RowSpan<'_, f32>,
) {
    let nb = cols / 32;
    let first = out.first();
    let xv = |n: usize| &x[n * x_stride..][..cols];
    for i in 0..out.len() {
        let r = first + i;
        let (q, s, b) = (
            &qs[r * cols / 2..][..cols / 2],
            &scales[r * nb..][..nb],
            &biases[r * nb..][..nb],
        );
        let bias = |n: usize| dot_bf16_f32(b, &xsum[n * xsum_stride..][..nb]);
        let mut n = 0;
        while n + 3 <= out.n_vec() {
            for (t, y) in q4_row(q, s, [xv(n), xv(n + 1), xv(n + 2)]).into_iter().enumerate() {
                out.vec(n + t)[i] = y + bias(n + t);
            }
            n += 3;
        }
        for n in n..out.n_vec() {
            let [y] = q4_row(q, s, [xv(n)]);
            out.vec(n)[i] = y + bias(n);
        }
    }
}
