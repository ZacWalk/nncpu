//! AVX2 + FMA kernels, written with `std::arch` intrinsics.
//!
//! Each follows the operation order of the llmcpu kernel it ports (the JIT
//! gemvs included): the same accumulators, fed in the same order, and the
//! same reduction trees. Rust never reassociates floating-point arithmetic,
//! so that order is what runs. Tails go through the `scalar::*_lane`
//! functions with a fused multiply-add, which gives a tail element the bits
//! a vector lane would.
//!
//! [`Avx2`] is a token that proves the CPU has AVX2 and FMA; the kernels are
//! safe methods on it that check their slices once, then run unchecked
//! pointer loops. Entry points are `#[inline(never)]` so that no AVX code is
//! inlined into the code that decides whether to call it.

use super::{KvRows, scalar};
use crate::pool::RowSpan;
use crate::runtime::bf16_to_f32;
use std::arch::x86_64::*;
use std::ops::Range;

/// Proof that this CPU and OS run AVX2 and FMA.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Avx2(());

impl Avx2 {
    pub fn detect() -> Option<Avx2> {
        crate::cpu::cpu_supported().then_some(Avx2(()))
    }
}

// ---- building blocks ----------------------------------------------------------

/// Sum of 8 lanes: high half onto low, then two `hadd`s.
#[inline]
#[target_feature(enable = "avx2,fma")]
fn hsum(v: __m256) -> f32 {
    let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
    let s = _mm_hadd_ps(s, s);
    _mm_cvtss_f32(_mm_hadd_ps(s, s))
}

/// Max of 8 lanes.
#[inline]
#[target_feature(enable = "avx2,fma")]
fn hmax(v: __m256) -> f32 {
    let m = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
    let m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    _mm_cvtss_f32(_mm_max_ss(m, _mm_shuffle_ps::<0x55>(m, m)))
}

/// `(a0 + a1) + (a2 + a3)`, then [`hsum`]: the four-accumulator reduction.
#[inline]
#[target_feature(enable = "avx2,fma")]
fn reduce4(a0: __m256, a1: __m256, a2: __m256, a3: __m256) -> f32 {
    hsum(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)))
}

/// One fused multiply-add, for the scalar tails.
#[inline]
#[target_feature(enable = "avx2,fma")]
fn fma1(a: f32, b: f32, c: f32) -> f32 {
    _mm_cvtss_f32(_mm_fmadd_ss(_mm_set_ss(a), _mm_set_ss(b), _mm_set_ss(c)))
}

/// 8 BF16 values widened to F32: zero-extend, then shift into the high half.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn load_bf16(p: *const u16) -> __m256 {
    // SAFETY: the caller guarantees 8 readable values at p.
    let h = unsafe { _mm_loadu_si128(p.cast()) };
    _mm256_castsi256_ps(_mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(h)))
}

/// A BF16 block scale in all 8 lanes. (A scalar shift and a broadcast:
/// cheaper than `vpbroadcastw` + `vpslld` as the intrinsics compile here.)
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn bf16_splat(p: *const u16) -> __m256 {
    // SAFETY: the caller guarantees p is readable.
    _mm256_set1_ps(bf16_to_f32(unsafe { *p }))
}

/// How far ahead of the current block the row kernels prefetch their
/// quants. The hardware prefetcher alone leaves a single thread that also
/// widens weights well short of DRAM bandwidth; measured on a Zen 3 this
/// adds 15-30% to one thread's streaming rate (4-16 KB ahead all do), and
/// prefetching past the end of a matrix is harmless.
const PREFETCH_AHEAD: usize = 4096;

#[inline]
#[target_feature(enable = "avx2,fma")]
fn prefetch<T>(p: *const T) {
    prefetch_at(p, PREFETCH_AHEAD);
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn prefetch_at<T>(p: *const T, ahead: usize) {
    // A prefetch is a hint that never faults, whatever the address. T1
    // (into L2) measures as well as T0, and is the one hint whose encoding
    // every rustc backend agrees on (one maps T0 to `prefetcht2`).
    // `_mm_prefetch` is a safe fn in current Rust but unsafe in the MSRV's.
    #[allow(unused_unsafe)]
    // SAFETY: prefetching does not access memory, so any address is sound.
    unsafe {
        _mm_prefetch::<_MM_HINT_T1>(p.cast::<u8>().wrapping_add(ahead).cast())
    }
}

/// AVX2 exp(x): x = n*ln(2) + r with r in [-ln(2)/2, ln(2)/2], a degree-5
/// minimax polynomial for exp(r), then 2^n added into the exponent field.
/// Max relative error ~3e-7 across [-87.3, 88.7]; outside that range it
/// saturates, which is what softmax wants (the negative side flushes to
/// ~0; the positive side cannot occur once the max is subtracted).
#[inline]
#[target_feature(enable = "avx2,fma")]
fn exp8(x: __m256) -> __m256 {
    use scalar::{EXP_HI, EXP_LO, EXP_P, LN2_HI, LN2_LO, LOG2E};
    let x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(EXP_LO)), _mm256_set1_ps(EXP_HI));
    let fx =
        _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(_mm256_mul_ps(x, _mm256_set1_ps(LOG2E)));
    let r = _mm256_fnmadd_ps(fx, _mm256_set1_ps(LN2_HI), x);
    let r = _mm256_fnmadd_ps(fx, _mm256_set1_ps(LN2_LO), r);
    let mut y = _mm256_fmadd_ps(_mm256_set1_ps(EXP_P[0]), r, _mm256_set1_ps(EXP_P[1]));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(EXP_P[2]));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(EXP_P[3]));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(EXP_P[4]));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(EXP_P[5]));
    y = _mm256_fmadd_ps(y, _mm256_mul_ps(r, r), r);
    y = _mm256_add_ps(y, _mm256_set1_ps(1.0));
    let e = _mm256_slli_epi32::<23>(_mm256_add_epi32(_mm256_cvtps_epi32(fx), _mm256_set1_epi32(0x7f)));
    _mm256_mul_ps(y, _mm256_castsi256_ps(e))
}

/// GELU(x) = 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))), tanh by a
/// Pade [7/6] approximant over [-5, 5] that saturates to sign(u) beyond:
///   tanh(u) ~= u (135135 + 17325 u^2 + 378 u^4 + u^6)
///             / (135135 + 62370 u^2 + 3150 u^4 + 28 u^6)
/// Under 5e-4 absolute error in the GELU output.
#[inline]
#[target_feature(enable = "avx2,fma")]
fn gelu8(v: __m256) -> __m256 {
    let set = _mm256_set1_ps;
    let v3 = _mm256_mul_ps(_mm256_mul_ps(v, v), v);
    let u = _mm256_mul_ps(set(scalar::GELU_C), _mm256_fmadd_ps(set(scalar::GELU_A), v3, v));
    let u2 = _mm256_mul_ps(u, u);
    let u4 = _mm256_mul_ps(u2, u2);
    let u6 = _mm256_mul_ps(u4, u2);
    let num = _mm256_add_ps(
        u6,
        _mm256_fmadd_ps(set(378.0), u4, _mm256_fmadd_ps(set(17325.0), u2, set(135135.0))),
    );
    let num = _mm256_mul_ps(u, num);
    let den = _mm256_fmadd_ps(
        set(28.0),
        u6,
        _mm256_fmadd_ps(set(3150.0), u4, _mm256_fmadd_ps(set(62370.0), u2, set(135135.0))),
    );
    let t_pade = _mm256_div_ps(num, den);
    let sign = set(-0.0);
    let sat = _mm256_cmp_ps::<_CMP_GE_OQ>(_mm256_andnot_ps(sign, u), set(5.0));
    let sign_u = _mm256_or_ps(_mm256_and_ps(u, sign), set(1.0));
    let t = _mm256_blendv_ps(t_pade, sign_u, sat);
    _mm256_mul_ps(_mm256_mul_ps(set(0.5), v), _mm256_add_ps(set(1.0), t))
}

/// sigmoid(g) = 1 / (1 + exp(0 - g)), a true divide (not `rcpps`).
#[inline]
#[target_feature(enable = "avx2,fma")]
fn sigmoid8(g: __m256) -> __m256 {
    let one = _mm256_set1_ps(1.0);
    _mm256_div_ps(one, _mm256_add_ps(one, exp8(_mm256_sub_ps(_mm256_setzero_ps(), g))))
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn exp1(x: f32) -> f32 {
    scalar::exp_lane(x, |a, b, c| fma1(a, b, c))
}

// ---- elementwise ------------------------------------------------------------------

impl Avx2 {
    pub fn gelu_inplace(self, x: &mut [f32]) {
        let p = x.as_mut_ptr();
        // SAFETY: `self` proves AVX2 + FMA; x holds x.len() floats, and the
        // kernel reads each element before writing it.
        unsafe { gelu_raw(p, p, x.len()) }
    }

    pub fn gelu_mul_inplace(self, gate: &mut [f32], up: &[f32]) {
        assert_eq!(gate.len(), up.len());
        // SAFETY: `self` proves AVX2 + FMA; both buffers hold gate.len() floats.
        unsafe { gelu_mul_raw(gate.as_mut_ptr(), up.as_ptr(), gate.len()) }
    }

    pub fn swiglu_inplace(self, gate: &mut [f32], up: &[f32]) {
        assert_eq!(gate.len(), up.len());
        let g = gate.as_mut_ptr();
        // SAFETY: as `gelu_inplace`.
        unsafe { swiglu_raw(g, g, up.as_ptr(), up.len()) }
    }

    pub fn silu_mul_into(self, up: &mut [f32], gate: &[f32]) {
        assert_eq!(gate.len(), up.len());
        let u = up.as_mut_ptr();
        // SAFETY: as `gelu_inplace`.
        unsafe { swiglu_raw(u, gate.as_ptr(), u, gate.len()) }
    }

    pub fn mul_sigmoid(self, y: &mut [f32], g: &[f32]) {
        assert_eq!(y.len(), g.len());
        // SAFETY: as `gelu`.
        unsafe { mul_sigmoid_raw(y.as_mut_ptr(), g.as_ptr(), y.len()) }
    }

    pub fn add_inplace(self, y: &mut [f32], b: &[f32]) {
        assert_eq!(y.len(), b.len());
        // SAFETY: as `gelu`.
        unsafe { add_raw(y.as_mut_ptr(), b.as_ptr(), y.len()) }
    }

    pub fn mul_inplace(self, y: &mut [f32], s: &[f32]) {
        assert_eq!(y.len(), s.len());
        // SAFETY: as `gelu`.
        unsafe { mul_raw(y.as_mut_ptr(), s.as_ptr(), y.len()) }
    }

    pub fn softmax_inplace(self, p: &mut [f32]) {
        // SAFETY: `self` proves AVX2 + FMA; p holds p.len() floats.
        unsafe { softmax_raw(p.as_mut_ptr(), p.len()) }
    }
}

#[inline(never)]
#[target_feature(enable = "avx2,fma")]
unsafe fn gelu_raw(y: *mut f32, x: *const f32, n: usize) {
    // SAFETY: x and y hold n floats; each chunk is read before it is written.
    unsafe {
        let mut i = 0;
        while i + 8 <= n {
            _mm256_storeu_ps(y.add(i), gelu8(_mm256_loadu_ps(x.add(i))));
            i += 8;
        }
        for i in i..n {
            *y.add(i) = scalar::gelu_lane(*x.add(i), |a, b, c| fma1(a, b, c));
        }
    }
}

#[inline(never)]
#[target_feature(enable = "avx2,fma")]
unsafe fn gelu_mul_raw(g: *mut f32, u: *const f32, n: usize) {
    // SAFETY: g and u hold n floats.
    unsafe {
        let mut i = 0;
        while i + 8 <= n {
            let r = _mm256_mul_ps(gelu8(_mm256_loadu_ps(g.add(i))), _mm256_loadu_ps(u.add(i)));
            _mm256_storeu_ps(g.add(i), r);
            i += 8;
        }
        for i in i..n {
            *g.add(i) = scalar::gelu_lane(*g.add(i), |a, b, c| fma1(a, b, c)) * *u.add(i);
        }
    }
}

/// y = silu(g) * u, where y is g or u.
#[inline(never)]
#[target_feature(enable = "avx2,fma")]
unsafe fn swiglu_raw(y: *mut f32, g: *const f32, u: *const f32, n: usize) {
    // SAFETY: all three hold n floats; each chunk is read before it is written.
    unsafe {
        let mut i = 0;
        while i + 8 <= n {
            let gv = _mm256_loadu_ps(g.add(i));
            let r = _mm256_mul_ps(_mm256_mul_ps(gv, sigmoid8(gv)), _mm256_loadu_ps(u.add(i)));
            _mm256_storeu_ps(y.add(i), r);
            i += 8;
        }
        for i in i..n {
            let gv = *g.add(i);
            *y.add(i) = (gv * scalar::sigmoid_lane(gv, |a, b, c| fma1(a, b, c))) * *u.add(i);
        }
    }
}

#[inline(never)]
#[target_feature(enable = "avx2,fma")]
unsafe fn mul_sigmoid_raw(y: *mut f32, g: *const f32, n: usize) {
    // SAFETY: y and g hold n floats.
    unsafe {
        let mut i = 0;
        while i + 8 <= n {
            let r = _mm256_mul_ps(_mm256_loadu_ps(y.add(i)), sigmoid8(_mm256_loadu_ps(g.add(i))));
            _mm256_storeu_ps(y.add(i), r);
            i += 8;
        }
        for i in i..n {
            *y.add(i) *= scalar::sigmoid_lane(*g.add(i), |a, b, c| fma1(a, b, c));
        }
    }
}

#[inline(never)]
#[target_feature(enable = "avx2,fma")]
unsafe fn add_raw(y: *mut f32, b: *const f32, n: usize) {
    // SAFETY: y and b hold n floats.
    unsafe {
        let mut i = 0;
        while i + 8 <= n {
            _mm256_storeu_ps(
                y.add(i),
                _mm256_add_ps(_mm256_loadu_ps(y.add(i)), _mm256_loadu_ps(b.add(i))),
            );
            i += 8;
        }
        for i in i..n {
            *y.add(i) += *b.add(i);
        }
    }
}

#[inline(never)]
#[target_feature(enable = "avx2,fma")]
unsafe fn mul_raw(y: *mut f32, s: *const f32, n: usize) {
    // SAFETY: y and s hold n floats.
    unsafe {
        let mut i = 0;
        while i + 8 <= n {
            _mm256_storeu_ps(
                y.add(i),
                _mm256_mul_ps(_mm256_loadu_ps(y.add(i)), _mm256_loadu_ps(s.add(i))),
            );
            i += 8;
        }
        for i in i..n {
            *y.add(i) *= *s.add(i);
        }
    }
}

#[inline(never)]
#[target_feature(enable = "avx2,fma")]
unsafe fn softmax_raw(p: *mut f32, n: usize) {
    if n == 0 {
        return;
    }
    // SAFETY: p holds n floats.
    unsafe {
        // Pass 1: the max.
        let mut m = f32::NEG_INFINITY;
        let mut i = 0;
        if n >= 8 {
            let mut vmax = _mm256_loadu_ps(p);
            i = 8;
            while i + 8 <= n {
                vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(p.add(i)));
                i += 8;
            }
            m = hmax(vmax);
        }
        for i in i..n {
            if *p.add(i) > m {
                m = *p.add(i);
            }
        }
        // Pass 2: exp(p - m) and the sum. exp8 clamps its argument to
        // ~ln(FLT_MIN), so -inf lanes would come out ~1e-38 rather than 0:
        // mask them, to honour the causal-mask contract.
        let vm = _mm256_set1_ps(m);
        let ninf = _mm256_set1_ps(f32::NEG_INFINITY);
        let mut vsum = _mm256_setzero_ps();
        let mut i = 0;
        while i + 8 <= n {
            let raw = _mm256_loadu_ps(p.add(i));
            let v = exp8(_mm256_sub_ps(raw, vm));
            let v = _mm256_andnot_ps(_mm256_cmp_ps::<_CMP_EQ_OQ>(raw, ninf), v);
            _mm256_storeu_ps(p.add(i), v);
            vsum = _mm256_add_ps(vsum, v);
            i += 8;
        }
        let mut sum = f64::from(hsum(vsum));
        for i in i..n {
            let v = &mut *p.add(i);
            if *v == f32::NEG_INFINITY {
                *v = 0.0;
            } else {
                *v = exp1(*v - m);
                sum += f64::from(*v);
            }
        }
        // Pass 3: scale by 1/sum. A fully masked row is already all zeros.
        let inv = if sum > 0.0 { (1.0 / sum) as f32 } else { 0.0 };
        let vinv = _mm256_set1_ps(inv);
        let mut i = 0;
        while i + 8 <= n {
            _mm256_storeu_ps(p.add(i), _mm256_mul_ps(_mm256_loadu_ps(p.add(i)), vinv));
            i += 8;
        }
        for i in i..n {
            *p.add(i) *= inv;
        }
    }
}

// ---- norms, dots and conversions -----------------------------------------------

impl Avx2 {
    pub fn rmsnorm(self, y: &mut [f32], x: &[f32], eps: f32) {
        assert_eq!(y.len(), x.len());
        // SAFETY: `self` proves AVX2 + FMA; both buffers hold y.len() floats.
        unsafe { rmsnorm_raw(y.as_mut_ptr(), x.as_ptr(), y.len(), eps) }
    }

    pub fn rmsnorm_inplace(self, x: &mut [f32], eps: f32) {
        let p = x.as_mut_ptr();
        // SAFETY: as `rmsnorm`; x is read before each element is written.
        unsafe { rmsnorm_raw(p, p, x.len(), eps) }
    }

    pub fn rmsnorm_gamma_multi(self, y: &mut [f32], x: &[f32], dim: usize, gamma: &[f32], eps: f32) {
        assert!(y.len() == x.len() && dim > 0 && x.len().is_multiple_of(dim) && gamma.len() == dim);
        // SAFETY: as `rmsnorm`; gamma holds dim floats.
        unsafe { rmsnorm_gamma_raw(y.as_mut_ptr(), x.as_ptr(), x.len() / dim, dim, gamma.as_ptr(), eps) }
    }

    pub fn rmsnorm_gamma_multi_inplace(self, x: &mut [f32], dim: usize, gamma: &[f32], eps: f32) {
        assert!(dim > 0 && x.len().is_multiple_of(dim) && gamma.len() == dim);
        let p = x.as_mut_ptr();
        // SAFETY: as `rmsnorm_inplace`; gamma holds dim floats.
        unsafe { rmsnorm_gamma_raw(p, p, x.len() / dim, dim, gamma.as_ptr(), eps) }
    }

    pub fn l2norm_multi(self, x: &mut [f32], dim: usize, eps: f32, scale: f32) {
        assert!(dim > 0 && x.len().is_multiple_of(dim));
        // SAFETY: `self` proves AVX2 + FMA; x holds x.len() floats.
        unsafe { l2norm_raw(x.as_mut_ptr(), x.len() / dim, dim, eps, scale) }
    }

    pub fn dot(self, a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        // SAFETY: `self` proves AVX2 + FMA; both hold a.len() floats.
        unsafe { dot_entry(a.as_ptr(), b.as_ptr(), a.len()) }
    }

    pub fn bf16_to_f32_row(self, src: &[u16], dst: &mut [f32]) {
        assert_eq!(src.len(), dst.len());
        // SAFETY: as `dot`.
        unsafe { bf16_row_raw(src.as_ptr(), dst.as_mut_ptr(), dst.len()) }
    }

    pub fn block_sums(self, x: &[f32], out: &mut [f32]) {
        assert_eq!(x.len(), out.len() * 32);
        // SAFETY: `self` proves AVX2 + FMA; x holds 32 floats per output.
        unsafe { block_sums_raw(x.as_ptr(), out.as_mut_ptr(), out.len()) }
    }
}

/// The norms' sum of squares: FMA lanes over whole 8-chunks (only when
/// there is one), reduced, then the tail in double.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn sum_sq(x: *const f32, n: usize) -> f64 {
    // SAFETY: x holds n floats.
    unsafe {
        let mut i = 0;
        let mut s = 0.0;
        if n >= 8 {
            let mut acc = _mm256_setzero_ps();
            while i + 8 <= n {
                let v = _mm256_loadu_ps(x.add(i));
                acc = _mm256_fmadd_ps(v, v, acc);
                i += 8;
            }
            s = f64::from(hsum(acc));
        }
        for i in i..n {
            let v = f64::from(*x.add(i));
            s += v * v;
        }
        s
    }
}

/// y = x * scale (y may be x).
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn scale_into(y: *mut f32, x: *const f32, n: usize, scale: f32) {
    // SAFETY: x and y hold n floats.
    unsafe {
        let vs = _mm256_set1_ps(scale);
        let mut i = 0;
        while i + 8 <= n {
            _mm256_storeu_ps(y.add(i), _mm256_mul_ps(_mm256_loadu_ps(x.add(i)), vs));
            i += 8;
        }
        for i in i..n {
            *y.add(i) = *x.add(i) * scale;
        }
    }
}

#[inline(never)]
#[target_feature(enable = "avx2,fma")]
unsafe fn rmsnorm_raw(y: *mut f32, x: *const f32, n: usize, eps: f32) {
    if n == 0 {
        return;
    }
    // SAFETY: x and y hold n floats.
    unsafe {
        let scale = (1.0 / (sum_sq(x, n) / n as f64 + f64::from(eps)).sqrt()) as f32;
        scale_into(y, x, n, scale);
    }
}

/// RMSNorm and gamma over `n_groups` runs of `dim`, gamma staying in L1.
#[inline(never)]
#[target_feature(enable = "avx2,fma")]
unsafe fn rmsnorm_gamma_raw(y: *mut f32, x: *const f32, n_groups: usize, dim: usize, gamma: *const f32, eps: f32) {
    let inv_dim = 1.0 / dim as f32;
    // SAFETY: x and y hold n_groups * dim floats, gamma dim.
    unsafe {
        for g in 0..n_groups {
            let (xg, yg) = (x.add(g * dim), y.add(g * dim));
            let scale = (1.0 / (sum_sq(xg, dim) * f64::from(inv_dim) + f64::from(eps)).sqrt()) as f32;
            let vs = _mm256_set1_ps(scale);
            let mut i = 0;
            while i + 8 <= dim {
                let v = _mm256_mul_ps(_mm256_loadu_ps(xg.add(i)), vs);
                _mm256_storeu_ps(yg.add(i), _mm256_mul_ps(v, _mm256_loadu_ps(gamma.add(i))));
                i += 8;
            }
            for i in i..dim {
                *yg.add(i) = *xg.add(i) * scale * *gamma.add(i);
            }
        }
    }
}

#[inline(never)]
#[target_feature(enable = "avx2,fma")]
unsafe fn l2norm_raw(x: *mut f32, n_groups: usize, dim: usize, eps: f32, scale: f32) {
    // SAFETY: x holds n_groups * dim floats.
    unsafe {
        for g in 0..n_groups {
            let xg = x.add(g * dim);
            let s = scale / (dot_raw(xg, xg, dim) + eps).sqrt();
            scale_into(xg, xg, dim, s);
        }
    }
}

/// Four accumulators over 32-element strides, leftover 8-chunks into the
/// first, `(s0 + s1) + (s2 + s3)`, then the scalar tail.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_raw(a: *const f32, b: *const f32, n: usize) -> f32 {
    // SAFETY: a and b hold n floats.
    unsafe {
        let mut s0 = _mm256_setzero_ps();
        let mut s1 = _mm256_setzero_ps();
        let mut s2 = _mm256_setzero_ps();
        let mut s3 = _mm256_setzero_ps();
        let mut i = 0;
        while i + 32 <= n {
            s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(i)), _mm256_loadu_ps(b.add(i)), s0);
            s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(i + 8)), _mm256_loadu_ps(b.add(i + 8)), s1);
            s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(i + 16)), _mm256_loadu_ps(b.add(i + 16)), s2);
            s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(i + 24)), _mm256_loadu_ps(b.add(i + 24)), s3);
            i += 32;
        }
        while i + 8 <= n {
            s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(i)), _mm256_loadu_ps(b.add(i)), s0);
            i += 8;
        }
        let mut acc = reduce4(s0, s1, s2, s3);
        for i in i..n {
            acc += *a.add(i) * *b.add(i);
        }
        acc
    }
}

#[inline(never)]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_entry(a: *const f32, b: *const f32, n: usize) -> f32 {
    // SAFETY: the caller's.
    unsafe { dot_raw(a, b, n) }
}

/// sum bf16(a[i]) * b[i]: one accumulator, then the scalar tail.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_bf16_raw(a: *const u16, b: *const f32, n: usize) -> f32 {
    // SAFETY: a and b hold n values.
    unsafe {
        let mut acc = _mm256_setzero_ps();
        let mut i = 0;
        while i + 8 <= n {
            acc = _mm256_fmadd_ps(load_bf16(a.add(i)), _mm256_loadu_ps(b.add(i)), acc);
            i += 8;
        }
        let mut s = hsum(acc);
        for i in i..n {
            s += bf16_to_f32(*a.add(i)) * *b.add(i);
        }
        s
    }
}

#[inline(never)]
#[target_feature(enable = "avx2,fma")]
unsafe fn bf16_row_raw(src: *const u16, dst: *mut f32, n: usize) {
    // SAFETY: src and dst hold n values.
    unsafe {
        let mut i = 0;
        while i + 8 <= n {
            _mm256_storeu_ps(dst.add(i), load_bf16(src.add(i)));
            i += 8;
        }
        for i in i..n {
            *dst.add(i) = bf16_to_f32(*src.add(i));
        }
    }
}

#[inline(never)]
#[target_feature(enable = "avx2,fma")]
unsafe fn block_sums_raw(x: *const f32, out: *mut f32, nb: usize) {
    // SAFETY: x holds 32 * nb floats, out nb.
    unsafe {
        for b in 0..nb {
            let xb = x.add(32 * b);
            let mut acc = _mm256_loadu_ps(xb);
            acc = _mm256_add_ps(acc, _mm256_loadu_ps(xb.add(8)));
            acc = _mm256_add_ps(acc, _mm256_loadu_ps(xb.add(16)));
            acc = _mm256_add_ps(acc, _mm256_loadu_ps(xb.add(24)));
            *out.add(b) = hsum(acc);
        }
    }
}

// ---- Gated DeltaNet ----------------------------------------------------------------

impl Avx2 {
    pub fn conv1d_silu(self, y: &mut [f32], x: &[f32], w: &[f32], n_tok: usize, ch: usize, k: usize) {
        let rows = if k == 0 || n_tok == 0 { 0 } else { k - 1 + n_tok };
        assert!(y.len() >= n_tok * ch && x.len() >= rows * ch && w.len() >= k * ch);
        // SAFETY: `self` proves AVX2 + FMA; the lengths are checked above.
        unsafe { conv1d_raw(y.as_mut_ptr(), x.as_ptr(), w.as_ptr(), n_tok, ch, k) }
    }

    pub fn delta_rule(
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
        assert!(s.len() >= d_v * d_k && q.len() >= d_k && k.len() >= d_k && v.len() >= d_v && o.len() >= d_v);
        // SAFETY: `self` proves AVX2 + FMA; the lengths are checked above.
        unsafe {
            delta_rule_raw(
                s.as_mut_ptr(),
                q.as_ptr(),
                k.as_ptr(),
                v.as_ptr(),
                decay,
                beta,
                o.as_mut_ptr(),
                d_k,
                d_v,
            )
        }
    }
}

#[inline(never)]
#[target_feature(enable = "avx2,fma")]
unsafe fn conv1d_raw(y: *mut f32, x: *const f32, w: *const f32, n_tok: usize, ch: usize, k: usize) {
    // SAFETY: y holds n_tok rows of ch floats, x k - 1 + n_tok, w k.
    unsafe {
        let one = _mm256_set1_ps(1.0);
        for t in 0..n_tok {
            let (xt, yt) = (x.add(t * ch), y.add(t * ch));
            let mut c = 0;
            while c + 8 <= ch {
                let mut acc = _mm256_setzero_ps();
                for j in 0..k {
                    acc = _mm256_fmadd_ps(
                        _mm256_loadu_ps(w.add(j * ch + c)),
                        _mm256_loadu_ps(xt.add(j * ch + c)),
                        acc,
                    );
                }
                let e = exp8(_mm256_sub_ps(_mm256_setzero_ps(), acc));
                _mm256_storeu_ps(yt.add(c), _mm256_div_ps(acc, _mm256_add_ps(one, e)));
                c += 8;
            }
            for c in c..ch {
                let mut acc = 0.0;
                for j in 0..k {
                    acc = fma1(*w.add(j * ch + c), *xt.add(j * ch + c), acc);
                }
                *yt.add(c) = acc / (1.0 + exp1(0.0 - acc));
            }
        }
    }
}

/// One token of the gated delta rule for one head: each state row is read,
/// updated and read again while it is in L1.
#[inline(never)]
#[target_feature(enable = "avx2,fma")]
unsafe fn delta_rule_raw(
    s: *mut f32,
    q: *const f32,
    k: *const f32,
    v: *const f32,
    decay: f32,
    beta: f32,
    o: *mut f32,
    d_k: usize,
    d_v: usize,
) {
    // SAFETY: s holds d_v rows of d_k floats, q and k d_k, v and o d_v.
    unsafe {
        let vdecay = _mm256_set1_ps(decay);
        for j in 0..d_v {
            let row = s.add(j * d_k);
            // The decayed row's projection on k, then the step towards v[j].
            let delta = beta * (*v.add(j) - decay * dot_raw(row, k, d_k));
            let vdelta = _mm256_set1_ps(delta);
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut i = 0;
            while i + 16 <= d_k {
                let s0 = _mm256_fmadd_ps(
                    vdelta,
                    _mm256_loadu_ps(k.add(i)),
                    _mm256_mul_ps(_mm256_loadu_ps(row.add(i)), vdecay),
                );
                let s1 = _mm256_fmadd_ps(
                    vdelta,
                    _mm256_loadu_ps(k.add(i + 8)),
                    _mm256_mul_ps(_mm256_loadu_ps(row.add(i + 8)), vdecay),
                );
                _mm256_storeu_ps(row.add(i), s0);
                _mm256_storeu_ps(row.add(i + 8), s1);
                acc0 = _mm256_fmadd_ps(s0, _mm256_loadu_ps(q.add(i)), acc0);
                acc1 = _mm256_fmadd_ps(s1, _mm256_loadu_ps(q.add(i + 8)), acc1);
                i += 16;
            }
            while i + 8 <= d_k {
                let s0 = _mm256_fmadd_ps(
                    vdelta,
                    _mm256_loadu_ps(k.add(i)),
                    _mm256_mul_ps(_mm256_loadu_ps(row.add(i)), vdecay),
                );
                _mm256_storeu_ps(row.add(i), s0);
                acc0 = _mm256_fmadd_ps(s0, _mm256_loadu_ps(q.add(i)), acc0);
                i += 8;
            }
            let mut out = hsum(_mm256_add_ps(acc0, acc1));
            for i in i..d_k {
                let s = *row.add(i) * decay + delta * *k.add(i);
                *row.add(i) = s;
                out += s * *q.add(i);
            }
            *o.add(j) = out;
        }
    }
}

// ---- attention -----------------------------------------------------------------------

/// A K/V cache element: F32, or BF16 widened like a BF16 weight.
trait Kv: Copy {
    const BF16: bool;
}

impl Kv for f32 {
    const BF16: bool = false;
}

impl Kv for u16 {
    const BF16: bool = true;
}

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn load8<T: Kv>(p: *const T) -> __m256 {
    // SAFETY: the caller guarantees 8 readable elements at p.
    unsafe {
        if T::BF16 {
            load_bf16(p.cast())
        } else {
            _mm256_loadu_ps(p.cast())
        }
    }
}

#[inline(always)]
unsafe fn load1<T: Kv>(p: *const T) -> f32 {
    // SAFETY: the caller guarantees p is readable.
    unsafe {
        if T::BF16 {
            bf16_to_f32(*p.cast::<u16>())
        } else {
            *p.cast::<f32>()
        }
    }
}

impl Avx2 {
    pub fn attn_block(
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
        super::check_attn(q, q_stride, k, v, kv_stride, n_keys, head_dim, n_heads, scores, part);
        let (q, s, p) = (q.as_ptr(), scores.as_mut_ptr(), part.as_mut_ptr());
        // SAFETY: `self` proves AVX2 + FMA; check_attn bounds every access.
        unsafe {
            match (k, v) {
                (KvRows::F32(k), KvRows::F32(v)) => attn_block_raw(
                    q,
                    q_stride,
                    k.as_ptr(),
                    v.as_ptr(),
                    kv_stride,
                    n_keys,
                    head_dim,
                    n_heads,
                    scale,
                    s,
                    p,
                ),
                (KvRows::Bf16(k), KvRows::Bf16(v)) => attn_block_raw(
                    q,
                    q_stride,
                    k.as_ptr(),
                    v.as_ptr(),
                    kv_stride,
                    n_keys,
                    head_dim,
                    n_heads,
                    scale,
                    s,
                    p,
                ),
                _ => unreachable!("check_attn rejects mixed K/V types"),
            }
        }
    }

    pub fn attn_merge(self, acc: &mut [f32], part: &[f32], head_dim: usize, n_heads: usize) {
        let n = n_heads * super::attn_part(head_dim);
        assert!(acc.len() >= n && part.len() >= n);
        // SAFETY: `self` proves AVX2 + FMA; both hold n_heads partials.
        unsafe { attn_merge_raw(acc.as_mut_ptr(), part.as_ptr(), head_dim, n_heads) }
    }
}

/// `attn_scores_N`: scores of key row `k` against N query heads at once, so
/// the row is loaded (and for BF16 widened) once for all of them. One named
/// accumulator per head keeps them in registers.
macro_rules! attn_scores {
    ($name:ident: $($g:literal => $acc:ident),+) => {
        #[inline]
        #[target_feature(enable = "avx2,fma")]
        unsafe fn $name<T: Kv>(
            q: *const f32,
            q_stride: usize,
            k: *const T,
            hd: usize,
            scale: f32,
            scores: *mut f32,
            n_keys: usize,
            t: usize,
        ) {
            // SAFETY: q holds the heads q_stride apart, k one row, scores
            // one row of n_keys per head.
            unsafe {
                $(let mut $acc = _mm256_setzero_ps();)+
                let hd8 = hd & !7;
                let mut i = 0;
                while i < hd8 {
                    let kv = load8(k.add(i));
                    $($acc = _mm256_fmadd_ps(_mm256_loadu_ps(q.add($g * q_stride + i)), kv, $acc);)+
                    i += 8;
                }
                $(
                    let mut s = hsum($acc);
                    for i in hd8..hd {
                        s += *q.add($g * q_stride + i) * load1(k.add(i));
                    }
                    *scores.add($g * n_keys + t) = s * scale;
                )+
            }
        }
    };
}

attn_scores!(attn_scores1: 0 => a0);
attn_scores!(attn_scores2: 0 => a0, 1 => a1);
attn_scores!(attn_scores3: 0 => a0, 1 => a1, 2 => a2);
attn_scores!(attn_scores4: 0 => a0, 1 => a1, 2 => a2, 3 => a3);

#[inline(never)]
#[target_feature(enable = "avx2,fma")]
unsafe fn attn_block_raw<T: Kv>(
    q: *const f32,
    q_stride: usize,
    kr: *const T,
    vr: *const T,
    kv_stride: usize,
    n_keys: usize,
    hd: usize,
    n_heads: usize,
    scale: f32,
    scores: *mut f32,
    part: *mut f32,
) {
    // SAFETY: the caller checked every buffer against these shapes.
    unsafe {
        let pw = hd + 2;
        let hd8 = hd & !7;
        for t in 0..n_keys {
            let k = kr.add(t * kv_stride);
            let mut h = 0;
            while h + 4 <= n_heads {
                attn_scores4(
                    q.add(h * q_stride),
                    q_stride,
                    k,
                    hd,
                    scale,
                    scores.add(h * n_keys),
                    n_keys,
                    t,
                );
                h += 4;
            }
            let (qh, sh) = (q.add(h * q_stride), scores.add(h * n_keys));
            match n_heads - h {
                3 => attn_scores3(qh, q_stride, k, hd, scale, sh, n_keys, t),
                2 => attn_scores2(qh, q_stride, k, hd, scale, sh, n_keys, t),
                1 => attn_scores1(qh, q_stride, k, hd, scale, sh, n_keys, t),
                _ => {}
            }
        }
        // Each head's weights exp(s - m), with m the block's own max.
        for h in 0..n_heads {
            let s = scores.add(h * n_keys);
            let mut m = *s;
            for t in 1..n_keys {
                if m < *s.add(t) {
                    m = *s.add(t);
                }
            }
            let vm = _mm256_set1_ps(m);
            let mut vsum = _mm256_setzero_ps();
            let mut t = 0;
            while t + 8 <= n_keys {
                let w = exp8(_mm256_sub_ps(_mm256_loadu_ps(s.add(t)), vm));
                _mm256_storeu_ps(s.add(t), w);
                vsum = _mm256_add_ps(vsum, w);
                t += 8;
            }
            let mut l = hsum(vsum);
            for t in t..n_keys {
                let w = exp1(*s.add(t) - m);
                *s.add(t) = w;
                l += w;
            }
            *part.add(h * pw + hd) = m;
            *part.add(h * pw + hd + 1) = l;
        }
        // Weighted sum of V rows, t outermost so each row is read once for
        // all heads; the first row writes, the rest accumulate.
        for t in 0..n_keys {
            let v = vr.add(t * kv_stride);
            for h in 0..n_heads {
                let w = *scores.add(h * n_keys + t);
                let vw = _mm256_set1_ps(w);
                let o = part.add(h * pw);
                let mut i = 0;
                if t == 0 {
                    while i < hd8 {
                        _mm256_storeu_ps(o.add(i), _mm256_mul_ps(vw, load8(v.add(i))));
                        i += 8;
                    }
                    for i in hd8..hd {
                        *o.add(i) = w * load1(v.add(i));
                    }
                } else {
                    while i < hd8 {
                        _mm256_storeu_ps(
                            o.add(i),
                            _mm256_fmadd_ps(vw, load8(v.add(i)), _mm256_loadu_ps(o.add(i))),
                        );
                        i += 8;
                    }
                    for i in hd8..hd {
                        *o.add(i) += w * load1(v.add(i));
                    }
                }
            }
        }
    }
}

#[inline(never)]
#[target_feature(enable = "avx2,fma")]
unsafe fn attn_merge_raw(acc: *mut f32, part: *const f32, hd: usize, n_heads: usize) {
    // SAFETY: acc and part hold n_heads partials of hd + 2 floats.
    unsafe {
        let pw = hd + 2;
        let hd8 = hd & !7;
        for h in 0..n_heads {
            let (a, b) = (acc.add(h * pw), part.add(h * pw));
            let (am, bm) = (*a.add(hd), *b.add(hd));
            let m = if am < bm { bm } else { am };
            let ca = (am - m).exp();
            let cb = (bm - m).exp();
            let (vca, vcb) = (_mm256_set1_ps(ca), _mm256_set1_ps(cb));
            let mut i = 0;
            while i < hd8 {
                let r = _mm256_fmadd_ps(
                    _mm256_loadu_ps(b.add(i)),
                    vcb,
                    _mm256_mul_ps(_mm256_loadu_ps(a.add(i)), vca),
                );
                _mm256_storeu_ps(a.add(i), r);
                i += 8;
            }
            for i in hd8..hd {
                *a.add(i) = *a.add(i) * ca + *b.add(i) * cb;
            }
            *a.add(hd) = m;
            *a.add(hd + 1) = *a.add(hd + 1) * ca + *b.add(hd + 1) * cb;
        }
    }
}

// ---- gemv --------------------------------------------------------------------------------

impl Avx2 {
    /// BF16 rows `out.first()..out.end()` (whole groups of 4) of every
    /// vector, with the 4-row kernel.
    pub fn gemv_bf16_quads(self, w: &[u16], x: &[f32], x_stride: usize, cols: usize, out: &mut RowSpan<'_, f32>) {
        assert!(cols.is_multiple_of(8) && out.first().is_multiple_of(4) && out.len().is_multiple_of(4));
        assert!(w.len() >= super::elems(out.end(), cols));
        assert!(x.len() >= super::strided_len(out.n_vec(), x_stride, cols));
        // SAFETY: `self` proves AVX2 + FMA; the rows and vectors read are checked above.
        unsafe { bf16_quads_raw(w.as_ptr(), x.as_ptr(), x_stride, cols, out) }
    }

    /// BF16 rows of one vector, with the 1-row kernel.
    pub fn gemv_bf16_rows(self, w: &[u16], x: &[f32], cols: usize, out: &mut RowSpan<'_, f32>) {
        assert!(cols.is_multiple_of(8) && out.n_vec() == 1 && x.len() >= cols);
        assert!(w.len() >= super::elems(out.end(), cols));
        // SAFETY: as `gemv_bf16_quads`.
        unsafe { bf16_rows_raw(w.as_ptr(), x.as_ptr(), cols, out) }
    }

    /// The largest of rows `4 * quads.start .. 4 * quads.end` and its index
    /// (the lowest on ties); None if no row beats -inf.
    pub fn gemv_bf16_argmax_quads(
        self,
        w: &[u16],
        x: &[f32],
        cols: usize,
        quads: Range<usize>,
    ) -> Option<(f32, usize)> {
        assert!(cols.is_multiple_of(8) && x.len() >= cols);
        assert!(w.len() >= super::elems(4 * quads.end, cols));
        // SAFETY: as `gemv_bf16_quads`.
        unsafe { bf16_argmax_raw(w.as_ptr(), x.as_ptr(), cols, quads) }
    }

    /// Q8_0 rows `out.first()..out.end()` of every vector: tokens three at a
    /// time, the rest one by one.
    pub fn gemv_q8_0_rows(
        self,
        qs: &[i8],
        scales: &[u16],
        x: &[f32],
        x_stride: usize,
        cols: usize,
        out: &mut RowSpan<'_, f32>,
    ) {
        assert!(cols.is_multiple_of(32));
        assert!(qs.len() >= super::elems(out.end(), cols) && scales.len() >= super::elems(out.end(), cols / 32));
        assert!(x.len() >= super::strided_len(out.n_vec(), x_stride, cols));
        // SAFETY: as `gemv_bf16_quads`.
        unsafe { q8_rows_raw(qs.as_ptr(), scales.as_ptr(), x.as_ptr(), x_stride, cols, out) }
    }

    /// Q4 rows of every vector, as `gemv_q8_0_rows`, each plus
    /// `dot(biases row, xsum)`.
    pub fn gemv_q4_rows(
        self,
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
        assert!(cols.is_multiple_of(32));
        let nb = super::elems(out.end(), cols / 32);
        assert!(qs.len() >= super::elems(out.end(), cols / 2) && scales.len() >= nb && biases.len() >= nb);
        assert!(x.len() >= super::strided_len(out.n_vec(), x_stride, cols));
        assert!(xsum.len() >= super::strided_len(out.n_vec(), xsum_stride, cols / 32));
        let (q, s, b) = (qs.as_ptr(), scales.as_ptr(), biases.as_ptr());
        // SAFETY: as `gemv_bf16_quads`.
        unsafe { q4_rows_raw(q, s, b, x.as_ptr(), x_stride, xsum.as_ptr(), xsum_stride, cols, out) }
    }
}

/// Four BF16 rows `cols` apart against one x: the x tile is loaded once per
/// step for the four row accumulators, which a 3-`hadd` tree and one add
/// reduce to the four sums.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn bf16_4rows(w: *const u16, cols: usize, x: *const f32) -> __m128 {
    // SAFETY: w holds 4 rows of cols values, x cols floats.
    unsafe {
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut a2 = _mm256_setzero_ps();
        let mut a3 = _mm256_setzero_ps();
        let (w1, w2, w3) = (w.add(cols), w.add(2 * cols), w.add(3 * cols));
        let quad_bytes = 8 * cols;
        let mut k = 0;
        while k < cols {
            if k.is_multiple_of(32) {
                // Each row is its own stream: fetch its place in the next
                // quad, one prefetch per row per 64 bytes.
                prefetch_at(w.add(k), quad_bytes);
                prefetch_at(w1.add(k), quad_bytes);
                prefetch_at(w2.add(k), quad_bytes);
                prefetch_at(w3.add(k), quad_bytes);
            }
            let xt = _mm256_loadu_ps(x.add(k));
            a0 = _mm256_fmadd_ps(load_bf16(w.add(k)), xt, a0);
            a1 = _mm256_fmadd_ps(load_bf16(w1.add(k)), xt, a1);
            a2 = _mm256_fmadd_ps(load_bf16(w2.add(k)), xt, a2);
            a3 = _mm256_fmadd_ps(load_bf16(w3.add(k)), xt, a3);
            k += 8;
        }
        let h = _mm256_hadd_ps(_mm256_hadd_ps(a0, a1), _mm256_hadd_ps(a2, a3));
        _mm_add_ps(_mm256_castps256_ps128(h), _mm256_extractf128_ps::<1>(h))
    }
}

/// One BF16 row: four accumulators when cols % 32 == 0, else one.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn bf16_row(w: *const u16, x: *const f32, cols: usize) -> f32 {
    // SAFETY: w and x hold cols values.
    unsafe {
        let mut a0 = _mm256_setzero_ps();
        let mut k = 0;
        if cols.is_multiple_of(32) {
            let mut a1 = _mm256_setzero_ps();
            let mut a2 = _mm256_setzero_ps();
            let mut a3 = _mm256_setzero_ps();
            while k < cols {
                a0 = _mm256_fmadd_ps(load_bf16(w.add(k)), _mm256_loadu_ps(x.add(k)), a0);
                a1 = _mm256_fmadd_ps(load_bf16(w.add(k + 8)), _mm256_loadu_ps(x.add(k + 8)), a1);
                a2 = _mm256_fmadd_ps(load_bf16(w.add(k + 16)), _mm256_loadu_ps(x.add(k + 16)), a2);
                a3 = _mm256_fmadd_ps(load_bf16(w.add(k + 24)), _mm256_loadu_ps(x.add(k + 24)), a3);
                k += 32;
            }
            reduce4(a0, a1, a2, a3)
        } else {
            while k < cols {
                a0 = _mm256_fmadd_ps(load_bf16(w.add(k)), _mm256_loadu_ps(x.add(k)), a0);
                k += 8;
            }
            hsum(a0)
        }
    }
}

#[inline(never)]
#[target_feature(enable = "avx2,fma")]
unsafe fn bf16_quads_raw(w: *const u16, x: *const f32, x_stride: usize, cols: usize, out: &mut RowSpan<'_, f32>) {
    let first = out.first();
    // SAFETY: the caller checked the rows and vectors against w and x.
    unsafe {
        for i in (0..out.len()).step_by(4) {
            let wq = w.add((first + i) * cols);
            for n in 0..out.n_vec() {
                let r = bf16_4rows(wq, cols, x.add(n * x_stride));
                _mm_storeu_ps(out.vec(n)[i..i + 4].as_mut_ptr(), r);
            }
        }
    }
}

#[inline(never)]
#[target_feature(enable = "avx2,fma")]
unsafe fn bf16_rows_raw(w: *const u16, x: *const f32, cols: usize, out: &mut RowSpan<'_, f32>) {
    let first = out.first();
    for (i, y) in out.vec(0).iter_mut().enumerate() {
        // SAFETY: the caller checked the rows against w.
        *y = unsafe { bf16_row(w.add((first + i) * cols), x, cols) };
    }
}

#[inline(never)]
#[target_feature(enable = "avx2,fma")]
unsafe fn bf16_argmax_raw(w: *const u16, x: *const f32, cols: usize, quads: Range<usize>) -> Option<(f32, usize)> {
    let mut best = None;
    let mut bv = f32::NEG_INFINITY;
    for q in quads {
        let r = 4 * q;
        let mut y = [0f32; 4];
        // SAFETY: the caller checked the rows against w.
        unsafe { _mm_storeu_ps(y.as_mut_ptr(), bf16_4rows(w.add(r * cols), cols, x)) };
        for (j, v) in y.into_iter().enumerate() {
            if v > bv {
                bv = v;
                best = Some((v, r + j));
            }
        }
    }
    best
}

/// 8 Q8_0 weights: sign-extended, converted and multiplied by the scale.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn q8_widen(q: *const i8, sc: __m256) -> __m256 {
    // SAFETY: the caller guarantees 8 readable bytes.
    let v = unsafe { _mm_loadl_epi64(q.cast()) };
    _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v)), sc)
}

/// One Q8_0 row: per block, the BF16 scale is broadcast and each group of 8
/// weights widened, multiplied by it and FMA'd against x into its own
/// accumulator. Four independent accumulators, one per FMA of a block: a
/// single one would chain the block's four 4-cycle FMAs and leave a thread
/// latency-bound well below what memory can feed.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn q8_row1(q: *const i8, s: *const u16, x: *const f32, nb: usize) -> f32 {
    // SAFETY: q and x hold 32 * nb values, s nb.
    unsafe {
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut a2 = _mm256_setzero_ps();
        let mut a3 = _mm256_setzero_ps();
        for b in 0..nb {
            let sc = bf16_splat(s.add(b));
            let (qb, xb) = (q.add(32 * b), x.add(32 * b));
            if b.is_multiple_of(2) {
                prefetch(qb);
            }
            a0 = _mm256_fmadd_ps(q8_widen(qb, sc), _mm256_loadu_ps(xb), a0);
            a1 = _mm256_fmadd_ps(q8_widen(qb.add(8), sc), _mm256_loadu_ps(xb.add(8)), a1);
            a2 = _mm256_fmadd_ps(q8_widen(qb.add(16), sc), _mm256_loadu_ps(xb.add(16)), a2);
            a3 = _mm256_fmadd_ps(q8_widen(qb.add(24), sc), _mm256_loadu_ps(xb.add(24)), a3);
        }
        reduce4(a0, a1, a2, a3)
    }
}

/// [`q8_row1`] for three tokens: each group of weights is widened and scaled
/// once and FMA'd into all three tokens' accumulators, in the 1-row order,
/// so every token's sum has the bits `q8_row1` gives it.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn q8_row3(q: *const i8, s: *const u16, x: [*const f32; 3], nb: usize) -> [f32; 3] {
    // SAFETY: q and each x hold 32 * nb values, s nb.
    unsafe {
        let z = _mm256_setzero_ps();
        let (mut a00, mut a01, mut a02, mut a03) = (z, z, z, z);
        let (mut a10, mut a11, mut a12, mut a13) = (z, z, z, z);
        let (mut a20, mut a21, mut a22, mut a23) = (z, z, z, z);
        let [x0, x1, x2] = x;
        for b in 0..nb {
            let sc = bf16_splat(s.add(b));
            let (qb, o) = (q.add(32 * b), 32 * b);
            if b.is_multiple_of(2) {
                prefetch(qb);
            }
            let w = q8_widen(qb, sc);
            a00 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x0.add(o)), a00);
            a10 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x1.add(o)), a10);
            a20 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x2.add(o)), a20);
            let w = q8_widen(qb.add(8), sc);
            a01 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x0.add(o + 8)), a01);
            a11 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x1.add(o + 8)), a11);
            a21 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x2.add(o + 8)), a21);
            let w = q8_widen(qb.add(16), sc);
            a02 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x0.add(o + 16)), a02);
            a12 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x1.add(o + 16)), a12);
            a22 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x2.add(o + 16)), a22);
            let w = q8_widen(qb.add(24), sc);
            a03 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x0.add(o + 24)), a03);
            a13 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x1.add(o + 24)), a13);
            a23 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x2.add(o + 24)), a23);
        }
        [
            reduce4(a00, a01, a02, a03),
            reduce4(a10, a11, a12, a13),
            reduce4(a20, a21, a22, a23),
        ]
    }
}

#[inline(never)]
#[target_feature(enable = "avx2,fma")]
unsafe fn q8_rows_raw(
    qs: *const i8,
    sc: *const u16,
    x: *const f32,
    x_stride: usize,
    cols: usize,
    out: &mut RowSpan<'_, f32>,
) {
    let nb = cols / 32;
    let (first, n_vec) = (out.first(), out.n_vec());
    // SAFETY: the caller checked the rows and vectors against qs, sc and x.
    unsafe {
        if n_vec == 1 {
            for (i, y) in out.vec(0).iter_mut().enumerate() {
                let r = first + i;
                *y = q8_row1(qs.add(r * cols), sc.add(r * nb), x, nb);
            }
            return;
        }
        let xv = |n: usize| x.add(n * x_stride);
        for i in 0..out.len() {
            let r = first + i;
            let (q, s) = (qs.add(r * cols), sc.add(r * nb));
            let mut n = 0;
            while n + 3 <= n_vec {
                for (t, y) in q8_row3(q, s, [xv(n), xv(n + 1), xv(n + 2)], nb).into_iter().enumerate() {
                    out.vec(n + t)[i] = y;
                }
                n += 3;
            }
            for n in n..n_vec {
                out.vec(n)[i] = q8_row1(q, s, xv(n), nb);
            }
        }
    }
}

/// A Q4 block's 16 bytes as two groups of 8, zero-extended to 32 bits.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn q4_bytes(q: *const u8) -> (__m256i, __m256i) {
    // SAFETY: the caller guarantees 16 readable bytes.
    unsafe {
        (
            _mm256_cvtepu8_epi32(_mm_loadl_epi64(q.cast())),
            _mm256_cvtepu8_epi32(_mm_loadl_epi64(q.add(8).cast())),
        )
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn q4_scale(v: __m256i, sc: __m256) -> __m256 {
    _mm256_mul_ps(_mm256_cvtepi32_ps(v), sc)
}

/// One Q4 row, without the bias term. Byte i of a block holds element i in
/// its low nibble and i + 16 in its high one, so the two byte groups give
/// elements 0..8 and 8..16 (low nibbles) and 16..24 and 24..32 (high
/// nibbles; the bytes were zero-extended, so a shift isolates them). Four
/// accumulators, as [`q8_row1`].
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn q4_row1(q: *const u8, s: *const u16, x: *const f32, nb: usize) -> f32 {
    // SAFETY: q holds 16 * nb bytes, x 32 * nb floats, s nb scales.
    unsafe {
        let mask = _mm256_set1_epi32(0x0f);
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut a2 = _mm256_setzero_ps();
        let mut a3 = _mm256_setzero_ps();
        for b in 0..nb {
            let sc = bf16_splat(s.add(b));
            let (lo, hi) = q4_bytes(q.add(16 * b));
            let xb = x.add(32 * b);
            if b.is_multiple_of(4) {
                prefetch(q.add(16 * b));
            }
            a0 = _mm256_fmadd_ps(q4_scale(_mm256_and_si256(lo, mask), sc), _mm256_loadu_ps(xb), a0);
            a1 = _mm256_fmadd_ps(q4_scale(_mm256_and_si256(hi, mask), sc), _mm256_loadu_ps(xb.add(8)), a1);
            a2 = _mm256_fmadd_ps(
                q4_scale(_mm256_srli_epi32::<4>(lo), sc),
                _mm256_loadu_ps(xb.add(16)),
                a2,
            );
            a3 = _mm256_fmadd_ps(
                q4_scale(_mm256_srli_epi32::<4>(hi), sc),
                _mm256_loadu_ps(xb.add(24)),
                a3,
            );
        }
        reduce4(a0, a1, a2, a3)
    }
}

/// [`q4_row1`] for three tokens, as [`q8_row3`]. Twelve accumulators, the
/// scale and the two byte groups leave one register, so low nibbles come
/// from a shift pair rather than a mask.
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn q4_row3(q: *const u8, s: *const u16, x: [*const f32; 3], nb: usize) -> [f32; 3] {
    // SAFETY: q holds 16 * nb bytes, each x 32 * nb floats, s nb scales.
    unsafe {
        let z = _mm256_setzero_ps();
        let (mut a00, mut a01, mut a02, mut a03) = (z, z, z, z);
        let (mut a10, mut a11, mut a12, mut a13) = (z, z, z, z);
        let (mut a20, mut a21, mut a22, mut a23) = (z, z, z, z);
        let [x0, x1, x2] = x;
        for b in 0..nb {
            let sc = bf16_splat(s.add(b));
            let (lo, hi) = q4_bytes(q.add(16 * b));
            let o = 32 * b;
            if b.is_multiple_of(4) {
                prefetch(q.add(16 * b));
            }
            let w = q4_scale(_mm256_srli_epi32::<28>(_mm256_slli_epi32::<28>(lo)), sc);
            a00 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x0.add(o)), a00);
            a10 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x1.add(o)), a10);
            a20 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x2.add(o)), a20);
            let w = q4_scale(_mm256_srli_epi32::<28>(_mm256_slli_epi32::<28>(hi)), sc);
            a01 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x0.add(o + 8)), a01);
            a11 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x1.add(o + 8)), a11);
            a21 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x2.add(o + 8)), a21);
            let w = q4_scale(_mm256_srli_epi32::<4>(lo), sc);
            a02 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x0.add(o + 16)), a02);
            a12 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x1.add(o + 16)), a12);
            a22 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x2.add(o + 16)), a22);
            let w = q4_scale(_mm256_srli_epi32::<4>(hi), sc);
            a03 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x0.add(o + 24)), a03);
            a13 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x1.add(o + 24)), a13);
            a23 = _mm256_fmadd_ps(w, _mm256_loadu_ps(x2.add(o + 24)), a23);
        }
        [
            reduce4(a00, a01, a02, a03),
            reduce4(a10, a11, a12, a13),
            reduce4(a20, a21, a22, a23),
        ]
    }
}

#[inline(never)]
#[target_feature(enable = "avx2,fma")]
unsafe fn q4_rows_raw(
    qs: *const u8,
    sc: *const u16,
    bi: *const u16,
    x: *const f32,
    x_stride: usize,
    xsum: *const f32,
    xsum_stride: usize,
    cols: usize,
    out: &mut RowSpan<'_, f32>,
) {
    let nb = cols / 32;
    let (first, n_vec) = (out.first(), out.n_vec());
    // SAFETY: the caller checked the rows and vectors against every buffer.
    unsafe {
        if n_vec == 1 {
            for (i, y) in out.vec(0).iter_mut().enumerate() {
                let r = first + i;
                *y = q4_row1(qs.add(r * cols / 2), sc.add(r * nb), x, nb) + dot_bf16_raw(bi.add(r * nb), xsum, nb);
            }
            return;
        }
        let xv = |n: usize| x.add(n * x_stride);
        for i in 0..out.len() {
            let r = first + i;
            let (q, s, b) = (qs.add(r * cols / 2), sc.add(r * nb), bi.add(r * nb));
            let bias = |n: usize| dot_bf16_raw(b, xsum.add(n * xsum_stride), nb);
            let mut n = 0;
            while n + 3 <= n_vec {
                for (t, y) in q4_row3(q, s, [xv(n), xv(n + 1), xv(n + 2)], nb).into_iter().enumerate() {
                    out.vec(n + t)[i] = y + bias(n + t);
                }
                n += 3;
            }
            for n in n..n_vec {
                out.vec(n)[i] = q4_row1(q, s, xv(n), nb) + bias(n);
            }
        }
    }
}
