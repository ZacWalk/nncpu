//! Kernel tests, ported from llmcpu's tests.cpp. Each SIMD kernel runs on
//! every backend this CPU has (portable, and AVX2 when available) against a
//! straightforward reference, with a relative tolerance: FMA rounding and
//! summation order differ from the reference, and that is expected.

use super::*;
use crate::runtime::{bf16_to_f32, f32_to_bf16};

/// xorshift64*: deterministic test data without dependencies.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// Uniform in [0, 1).
    fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }

    fn normal(&mut self) -> f32 {
        let u1 = self.unit().max(1e-7);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }

    fn vec(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..n).map(|_| self.uniform(lo, hi)).collect()
    }

    fn normals(&mut self, n: usize, sd: f32) -> Vec<f32> {
        (0..n).map(|_| sd * self.normal()).collect()
    }

    fn bf16s(&mut self, n: usize, lo: f32, hi: f32) -> Vec<u16> {
        (0..n).map(|_| f32_to_bf16(self.uniform(lo, hi))).collect()
    }
}

/// llmcpu's tolerance: relative above 1, absolute below.
fn close(a: f32, b: f32, rel: f32) -> bool {
    (a - b).abs() <= rel * a.abs().max(b.abs()).max(1.0)
}

#[track_caller]
fn assert_close(got: &[f32], want: &[f32], rel: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(close(g, w, rel), "{what}: [{i}] got {g}, want {w}");
    }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn dot_f64(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(&a, &b)| f64::from(a) * f64::from(b)).sum()
}

fn gelu_ref(v: f32) -> f32 {
    let u = scalar::GELU_C * (v + scalar::GELU_A * v * v * v);
    0.5 * v * (1.0 + u.tanh())
}

fn sigmoid_ref(g: f32) -> f32 {
    1.0 / (1.0 + (-g).exp())
}

// ---- CPU ----------------------------------------------------------------------------

#[test]
fn cpu_features_are_consistent() {
    let f = crate::cpu::cpu_features();
    assert_eq!(crate::cpu::cpu_supported(), f.avx2 && f.fma);
    if crate::cpu::simd() {
        assert!(crate::cpu::cpu_supported());
    }
    #[cfg(target_arch = "x86_64")]
    {
        assert_eq!(f.avx2, std::arch::is_x86_feature_detected!("avx2"));
        assert_eq!(f.fma, std::arch::is_x86_feature_detected!("fma"));
        if cfg!(all(target_feature = "avx2", target_feature = "fma")) {
            assert!(crate::cpu::cpu_supported());
        }
        assert_eq!(matches!(Backend::current(), Backend::Avx2(_)), crate::cpu::simd());
    }
}

// ---- activations ------------------------------------------------------------------------

#[test]
fn gelu_matches_reference() {
    let mut rng = Rng::new(0xdead);
    for be in Backend::all() {
        for n in [1024usize, 1, 7, 13, 1031] {
            let x = rng.vec(n, -5.0, 5.0);
            let mut y = x.clone();
            be.gelu_inplace(&mut y);
            let want: Vec<f32> = x.iter().map(|&v| gelu_ref(v)).collect();
            assert_close(&y, &want, 5e-4, &format!("{be:?} gelu n={n}"));
        }
        let mut out = [0.0f32, 1.0, -1.0, 3.0, -3.0, 6.0, -6.0, 100.0, -100.0];
        be.gelu_inplace(&mut out);
        assert!(close(out[0], 0.0, 1e-6));
        assert!(close(out[1], 0.841192, 1e-4), "{be:?} GELU(1) = {}", out[1]);
        assert!(close(out[2], -0.158808, 1e-4), "{be:?} GELU(-1) = {}", out[2]);
        assert_eq!(out[7], 100.0);
        assert_eq!(out[8], -0.0);
    }
}

#[test]
fn gated_activations_match_reference() {
    let mut rng = Rng::new(0x5167);
    for be in Backend::all() {
        for n in [257usize, 8, 5] {
            let (g, u) = (rng.vec(n, -3.0, 3.0), rng.vec(n, -3.0, 3.0));
            let silu: Vec<f32> = g.iter().zip(&u).map(|(&g, &u)| g * sigmoid_ref(g) * u).collect();
            let mut y = g.clone();
            be.swiglu_inplace(&mut y, &u);
            assert_close(&y, &silu, 1e-5, &format!("{be:?} swiglu n={n}"));
            // up = silu(gate) * up: the same function with the other operand in place.
            let mut z = u.clone();
            be.silu_mul_into(&mut z, &g);
            assert_eq!(bits(&z), bits(&y), "{be:?} silu_mul_into");
            let mut y = g.clone();
            be.gelu_mul_inplace(&mut y, &u);
            let want: Vec<f32> = g.iter().zip(&u).map(|(&g, &u)| gelu_ref(g) * u).collect();
            assert_close(&y, &want, 5e-4, &format!("{be:?} gelu_mul n={n}"));
        }
    }
}

#[test]
fn mul_sigmoid_matches_reference() {
    let mut rng = Rng::new(11);
    for be in Backend::all() {
        for n in [1usize, 7, 8, 37] {
            let (y0, g) = (rng.vec(n, -12.0, 12.0), rng.vec(n, -12.0, 12.0));
            let want: Vec<f32> = y0.iter().zip(&g).map(|(&y, &g)| y / (1.0 + (-g).exp())).collect();
            let mut y = y0.clone();
            be.mul_sigmoid(&mut y, &g);
            assert_close(&y, &want, 1e-5, &format!("{be:?} mul_sigmoid n={n}"));
        }
    }
}

#[test]
fn elementwise_add_and_mul() {
    let mut rng = Rng::new(3);
    for be in Backend::all() {
        for n in [0usize, 3, 8, 21] {
            let (a, b) = (rng.vec(n, -2.0, 2.0), rng.vec(n, -2.0, 2.0));
            let mut y = a.clone();
            be.add_inplace(&mut y, &b);
            assert!(y.iter().zip(a.iter().zip(&b)).all(|(&y, (&a, &b))| y == a + b));
            let mut y = a.clone();
            be.mul_inplace(&mut y, &b);
            assert!(y.iter().zip(a.iter().zip(&b)).all(|(&y, (&a, &b))| y == a * b));
        }
    }
}

#[test]
fn softcap_saturates() {
    let cap = 30.0;
    let input = [0.0f32, 1.0, -1.0, 5.0, -5.0, 30.0, 100.0, -100.0, 1e6];
    let mut out = input;
    softcap_inplace(&mut out, cap);
    for (&o, &x) in out.iter().zip(&input) {
        assert!(close(o, (x / cap).tanh() * cap, 1e-5));
        assert!(o.abs() <= cap + 1e-4);
    }
    assert_eq!(out[0], 0.0);
    assert!(out[8] >= cap - 1e-3 && out[8] <= cap);
}

// ---- softmax -------------------------------------------------------------------------------

#[test]
fn softmax_known_values_and_masks() {
    let ninf = f32::NEG_INFINITY;
    for be in Backend::all() {
        let mut p = [1.0f32, 2.0, 3.0, 4.0];
        be.softmax_inplace(&mut p);
        let e: Vec<f64> = (1..=4).map(|i| f64::from(i as f32 - 4.0).exp()).collect();
        let s: f64 = e.iter().sum();
        for (i, &v) in p.iter().enumerate() {
            assert!(close(v, (e[i] / s) as f32, 1e-6), "{be:?} softmax [{i}] = {v}");
        }
        // -inf masks: those entries become 0 and the rest sum to 1.
        let mut q = [0.5, ninf, 1.5, ninf, 2.5];
        be.softmax_inplace(&mut q);
        assert!(q[1] == 0.0 && q[3] == 0.0);
        assert!(close(q.iter().sum(), 1.0, 1e-6));
        assert!(q[0] < q[2] && q[2] < q[4]);
        // The same through the 8-wide path.
        let mut w: Vec<f32> = (0..19).map(|i| if i % 3 == 0 { ninf } else { i as f32 }).collect();
        be.softmax_inplace(&mut w);
        assert!(w.iter().enumerate().all(|(i, &v)| i % 3 != 0 || v == 0.0));
        assert!((w.iter().map(|&v| f64::from(v)).sum::<f64>() - 1.0).abs() < 1e-6);
        // A fully masked row is all zeros, not NaN.
        let mut m = [ninf; 11];
        be.softmax_inplace(&mut m);
        assert!(m.iter().all(|&v| v == 0.0));
        be.softmax_inplace(&mut []);
        // Random: non-negative, sums to 1, close to the exact softmax.
        let mut rng = Rng::new(0xc0ffee);
        let v0 = rng.vec(1024, -10.0, 10.0);
        let mut v = v0.clone();
        be.softmax_inplace(&mut v);
        let mx = v0.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let e: Vec<f64> = v0.iter().map(|&x| f64::from(x - mx).exp()).collect();
        let s: f64 = e.iter().sum();
        assert!(v.iter().all(|&x| x >= 0.0));
        assert!((v.iter().map(|&x| f64::from(x)).sum::<f64>() - 1.0).abs() < 1e-5);
        for (i, (&got, &e)) in v.iter().zip(&e).enumerate() {
            assert!(
                (f64::from(got) - e / s).abs() <= 2e-6 * (e / s) + 1e-12,
                "{be:?} softmax [{i}]"
            );
        }
    }
}

// ---- norms -------------------------------------------------------------------------------------

#[test]
fn rmsnorm_matches_reference() {
    let mut rng = Rng::new(0xcafe);
    for be in Backend::all() {
        // Gemma E2B/E4B hidden sizes and an odd size for the tail.
        for n in [1536usize, 2560, 256, 13] {
            let x = rng.vec(n, -3.0, 3.0);
            let mut y = vec![0.0; n];
            be.rmsnorm(&mut y, &x, 1e-6);
            let scale = (1.0 / (dot_f64(&x, &x) / n as f64 + 1e-6).sqrt()) as f32;
            let want: Vec<f32> = x.iter().map(|&v| v * scale).collect();
            assert_close(&y, &want, 1e-5, &format!("{be:?} rmsnorm n={n}"));
            let rms = (dot_f64(&y, &y) / n as f64).sqrt();
            assert!((rms - 1.0).abs() < 1e-3);
            let mut z = x.clone();
            be.rmsnorm_inplace(&mut z, 1e-6);
            assert_eq!(bits(&z), bits(&y), "{be:?} rmsnorm in place");
        }
    }
}

#[test]
fn rmsnorm_gamma_multi_matches_reference() {
    let mut rng = Rng::new(0x9a11);
    for be in Backend::all() {
        for (groups, dim) in [(3usize, 256usize), (5, 37), (4, 8), (2, 3)] {
            let x = rng.vec(groups * dim, -3.0, 3.0);
            let gamma = rng.vec(dim, 0.5, 1.5);
            let mut y = vec![0.0; groups * dim];
            be.rmsnorm_gamma_multi(&mut y, &x, dim, &gamma, 1e-6);
            for g in 0..groups {
                let xg = &x[g * dim..][..dim];
                let scale = 1.0 / (dot_f64(xg, xg) / dim as f64 + 1e-6).sqrt();
                let want: Vec<f32> = xg
                    .iter()
                    .zip(&gamma)
                    .map(|(&v, &k)| (f64::from(v) * scale) as f32 * k)
                    .collect();
                assert_close(
                    &y[g * dim..][..dim],
                    &want,
                    1e-5,
                    &format!("{be:?} rmsnorm_gamma {groups}x{dim}"),
                );
            }
            let mut z = x.clone();
            be.rmsnorm_gamma_multi_inplace(&mut z, dim, &gamma, 1e-6);
            assert_eq!(bits(&z), bits(&y), "{be:?} rmsnorm_gamma in place");
        }
    }
}

#[test]
fn l2norm_multi_matches_reference() {
    let mut rng = Rng::new(12);
    for be in Backend::all() {
        let (groups, dim, eps, scale) = (3, 37, 1e-6f32, 0.5f32);
        let x0 = rng.normals(groups * dim, 2.0);
        let mut want = vec![0.0; x0.len()];
        for g in 0..groups {
            let xg = &x0[g * dim..][..dim];
            let ss = dot_f64(xg, xg);
            for i in 0..dim {
                want[g * dim + i] = (f64::from(xg[i]) * f64::from(scale) / (ss + f64::from(eps)).sqrt()) as f32;
            }
        }
        let mut x = x0.clone();
        be.l2norm_multi(&mut x, dim, eps, scale);
        assert_close(&x, &want, 1e-5, &format!("{be:?} l2norm_multi"));
    }
}

// ---- dots and conversions ------------------------------------------------------------------

#[test]
fn dot_matches_reference() {
    let mut rng = Rng::new(0xc0ffee);
    for be in Backend::all() {
        for n in [8usize, 64, 768, 4096, 0, 1, 7, 13, 33, 100, 1000] {
            let (a, b) = (rng.vec(n, -1.0, 1.0), rng.vec(n, -1.0, 1.0));
            let got = be.dot(&a, &b);
            assert!(close(got, dot_f64(&a, &b) as f32, 1e-4), "{be:?} dot n={n}");
        }
    }
}

/// llmcpu's JIT gemv_f32 test, for the F32 matrices the model multiplies a
/// row at a time with `dot` (routers, Gated DeltaNet gates).
#[test]
fn f32_rows_by_dot_match_reference() {
    let mut rng = Rng::new(0xbeef);
    for be in Backend::all() {
        for (rows, cols) in [(1usize, 8usize), (4, 64), (16, 128), (32, 768), (3, 13), (5, 100)] {
            let (w, x) = (rng.vec(rows * cols, -1.0, 1.0), rng.vec(cols, -1.0, 1.0));
            for r in 0..rows {
                let row = &w[r * cols..][..cols];
                assert!(
                    close(be.dot(row, &x), dot_f64(row, &x) as f32, 1e-4),
                    "{be:?} {rows}x{cols} row {r}"
                );
            }
        }
    }
}

#[test]
fn bf16_to_f32_row_is_exact() {
    let mut rng = Rng::new(0xbf16abcd);
    let src = rng.bf16s(137, -100.0, 100.0); // not a multiple of 8: the tail too
    for be in Backend::all() {
        let mut got = vec![0.0; src.len()];
        be.bf16_to_f32_row(&src, &mut got);
        assert!(
            got.iter().zip(&src).all(|(&g, &s)| g.to_bits() == u32::from(s) << 16),
            "{be:?}"
        );
    }
}

#[test]
fn block_sums_match_reference() {
    let mut rng = Rng::new(0xb10c);
    let x = rng.vec(128, -4.0, 4.0);
    for be in Backend::all() {
        let mut out = [0.0; 4];
        be.block_sums(&x, &mut out);
        for (b, &o) in out.iter().enumerate() {
            let want: f64 = x[32 * b..][..32].iter().map(|&v| f64::from(v)).sum();
            assert!(close(o, want as f32, 1e-5), "{be:?} block {b}");
        }
    }
}

// ---- gemv -------------------------------------------------------------------------------------------

fn gemv_ref(w: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    (0..rows).map(|r| dot_f64(&w[r * cols..][..cols], x) as f32).collect()
}

fn q8_dequant(qs: &[i8], scales: &[u16]) -> Vec<f32> {
    qs.iter()
        .enumerate()
        .map(|(i, &q)| bf16_to_f32(scales[i / 32]) * f32::from(q))
        .collect()
}

fn q4_dequant(qs: &[u8], scales: &[u16], biases: &[u16]) -> Vec<f32> {
    let mut w = vec![0.0; qs.len() * 2];
    for (b, blk) in qs.as_chunks::<16>().0.iter().enumerate() {
        let (sc, bi) = (bf16_to_f32(scales[b]), bf16_to_f32(biases[b]));
        for k in 0..16 {
            w[32 * b + k] = sc * f32::from(blk[k] & 0x0f) + bi;
            w[32 * b + k + 16] = sc * f32::from(blk[k] >> 4) + bi;
        }
    }
    w
}

/// A quantized test matrix, in both formats.
struct QMat {
    q8: (Vec<i8>, Vec<u16>),
    q4: (Vec<u8>, Vec<u16>, Vec<u16>),
}

fn quantized(w: &[f32], rows: usize, cols: usize) -> QMat {
    let (n, nb) = (rows * cols, rows * cols / 32);
    // Q8_0 from the BF16-rounded weights.
    let wb: Vec<f32> = w.iter().map(|&v| bf16_to_f32(f32_to_bf16(v))).collect();
    let mut q8 = (vec![0i8; n], vec![0u16; nb]);
    quantize_f32_to_q8_0(&wb, &mut q8.0, &mut q8.1);
    let mut q4 = (vec![0u8; n / 2], vec![0u16; nb], vec![0u16; nb]);
    for r in 0..rows {
        let rb = r * cols / 32;
        quantize_f32_to_q4(
            &w[r * cols..][..cols],
            &mut q4.0[r * cols / 2..][..cols / 2],
            &mut q4.1[rb..][..cols / 32],
            &mut q4.2[rb..][..cols / 32],
        );
    }
    QMat { q8, q4 }
}

fn block_sums_of(x: &[f32], n_vec: usize, cols: usize) -> Vec<f32> {
    let mut xs = vec![0.0; n_vec * cols / 32];
    for t in 0..n_vec {
        block_sums(&x[t * cols..][..cols], &mut xs[t * cols / 32..][..cols / 32]);
    }
    xs
}

#[test]
fn gemv_bf16_matches_reference() {
    let mut rng = Rng::new(0xb16e);
    // Gemma E2B/E4B shapes, plus small ones off the fast paths: cols % 8 != 0
    // (f64 fallback), rows % 4 != 0 (1-row kernel, with and without cols % 32).
    let shapes = [
        (1usize, 8usize),
        (7, 5),
        (16, 1536),
        (1536, 1536),
        (2048, 1536),
        (6144, 1536),
        (6, 64),
        (5, 24),
        (260, 512),
    ];
    for (rows, cols) in shapes {
        let w = rng.bf16s(rows * cols, -1.0, 1.0);
        let wf: Vec<f32> = w.iter().map(|&v| bf16_to_f32(v)).collect();
        let x = rng.vec(cols, -1.0, 1.0);
        let want = gemv_ref(&wf, &x, rows, cols);
        for be in Backend::all() {
            let mut y = vec![0.0; rows];
            be.gemv_bf16(&w, &x, &mut y, rows, cols);
            assert_close(&y, &want, 1e-3, &format!("{be:?} gemv_bf16 {rows}x{cols}"));
        }
    }
}

#[test]
fn gemv_bf16_argmax_matches_gemv() {
    let mut rng = Rng::new(0xa12a);
    // The streaming path (rows % 4 == 0, cols % 8 == 0), threaded and not,
    // and the materialising fallback.
    for (rows, cols) in [(4usize, 32usize), (16, 256), (2048, 256), (256, 1536), (7, 5), (6, 64)] {
        let w = rng.bf16s(rows * cols, -1.0, 1.0);
        let wf: Vec<f32> = w.iter().map(|&v| bf16_to_f32(v)).collect();
        let x = rng.vec(cols, -1.0, 1.0);
        let want = gemv_ref(&wf, &x, rows, cols);
        let ref_best = (1..rows).fold(0, |b, r| if want[r] > want[b] { r } else { b });
        for be in Backend::all() {
            let got = be.gemv_bf16_argmax(&w, &x, rows, cols);
            // FMA order may flip a near tie; then the values must be close.
            assert!(
                close(want[got], want[ref_best], 1e-3),
                "{be:?} argmax {rows}x{cols}: {got} vs {ref_best}"
            );
            // Against the gemv itself it is exact: the same kernel, ties to the lower row.
            let mut y = vec![0.0; rows];
            be.gemv_bf16(&w, &x, &mut y, rows, cols);
            let best = (1..rows).fold(0, |b, r| if y[r] > y[b] { r } else { b });
            assert_eq!(got, best, "{be:?} argmax vs gemv {rows}x{cols}");
        }
    }
    // Ties go to the lower row; all-NaN gives 0.
    let (rows, cols) = (512, 256);
    let mut w = vec![0u16; rows * cols];
    for r in [100, 300, 301] {
        w[r * cols] = f32_to_bf16(1.0);
    }
    let mut x = vec![0.0; cols];
    x[0] = 2.0;
    for be in Backend::all() {
        assert_eq!(be.gemv_bf16_argmax(&w, &x, rows, cols), 100, "{be:?} tie");
        x[0] = f32::NAN;
        assert_eq!(be.gemv_bf16_argmax(&w, &x, rows, cols), 0, "{be:?} NaN");
        x[0] = 2.0;
    }
}

#[test]
fn q8_0_quantize_roundtrip() {
    let mut rng = Rng::new(0xc0ffee);
    let (rows, cols) = (3, 64);
    let w = rng.bf16s(rows * cols, -2.5, 2.5);
    let wf: Vec<f32> = w.iter().map(|&v| bf16_to_f32(v)).collect();
    let (mut qs, mut scales) = (vec![0i8; rows * cols], vec![0u16; rows * cols / 32]);
    quantize_f32_to_q8_0(&wf, &mut qs, &mut scales);
    for (i, (&q, &w)) in qs.iter().zip(&w).enumerate() {
        let scale = bf16_to_f32(scales[i / 32]);
        assert!(
            (scale * f32::from(q) - bf16_to_f32(w)).abs() <= scale + 1e-6,
            "element {i}"
        );
        assert!((-127..=127).contains(&q));
    }
    // The block max maps to +-127 against its BF16-rounded scale.
    let mut src = [0.0f32; 32];
    src[5] = -3.0;
    src[9] = 1.5;
    let (mut q, mut s) = ([0i8; 32], [0u16; 1]);
    quantize_f32_to_q8_0(&src, &mut q, &mut s);
    assert_eq!(bf16_to_f32(s[0]), bf16_to_f32(f32_to_bf16(3.0 / 127.0)));
    assert_eq!((q[5], q[0]), (-127, 0));
    // An all-zero block: scale 0, quants 0.
    quantize_f32_to_q8_0(&[0.0; 32], &mut q, &mut s);
    assert!(s[0] == 0 && q.iter().all(|&q| q == 0));
}

#[test]
fn rint_rounds_half_to_even() {
    for v in [
        0.5f32,
        1.5,
        2.5,
        -0.5,
        -1.5,
        -2.5,
        126.5,
        -126.5,
        0.49999997,
        3.7,
        -3.7,
        1e9,
        -1e9,
        4_194_304.5,
        8e6 + 0.5,
    ] {
        assert_eq!(quant::rint(v), v.round_ties_even() as i32, "{v}");
    }
    assert_eq!(quant::rint(f32::NAN), 0);
}

#[test]
fn gemv_q8_0_matches_reference() {
    let mut rng = Rng::new(0xc8e2);
    for (rows, cols) in [
        (1usize, 32usize),
        (8, 64),
        (64, 128),
        (256, 256),
        (512, 512),
        (2048, 2048),
    ] {
        let w = rng.vec(rows * cols, -1.0, 1.0);
        let m = quantized(&w, rows, cols);
        let x = rng.vec(cols, -1.0, 1.0);
        let want = gemv_ref(&q8_dequant(&m.q8.0, &m.q8.1), &x, rows, cols);
        for be in Backend::all() {
            let mut y = vec![0.0; rows];
            be.gemv_q8_0(&m.q8.0, &m.q8.1, &x, &mut y, rows, cols);
            assert_close(&y, &want, 1e-3, &format!("{be:?} gemv_q8_0 {rows}x{cols}"));
        }
    }
}

#[test]
fn q4_quantize_roundtrip() {
    // Scale and bias exact in BF16 and values on the 16 levels: the round
    // trip is exact, which pins down the nibble packing.
    let mut src = [0.0f32; 64];
    for i in 0..32 {
        src[i] = -2.0 + (i % 16) as f32 * 0.25;
        src[32 + i] = 4.0 + (i % 16) as f32 * 0.5;
    }
    let (mut qs, mut sc, mut bi) = ([0u8; 32], [0u16; 2], [0u16; 2]);
    quantize_f32_to_q4(&src, &mut qs, &mut sc, &mut bi);
    for (b, (s, bias)) in [(0.25f32, -2.0f32), (0.5, 4.0)].into_iter().enumerate() {
        assert_eq!((bf16_to_f32(sc[b]), bf16_to_f32(bi[b])), (s, bias));
        for k in 0..16 {
            let byte = qs[16 * b + k];
            // Byte k holds element k low, element k + 16 high.
            assert_eq!(s * f32::from(byte & 0x0f) + bias, src[32 * b + k]);
            assert_eq!(s * f32::from(byte >> 4) + bias, src[32 * b + k + 16]);
        }
    }
    // Arbitrary values: half a step, plus the BF16 rounding of scale and
    // bias amplified by the largest level.
    let mut rng = Rng::new(0x4b15);
    let src = rng.vec(32, -3.0, 7.0);
    let (mut qs, mut sc, mut bi) = ([0u8; 16], [0u16; 1], [0u16; 1]);
    quantize_f32_to_q4(&src, &mut qs, &mut sc, &mut bi);
    let (s, bias) = (bf16_to_f32(sc[0]), bf16_to_f32(bi[0]));
    let tol = 0.5 * s + (15.0 * s + bias.abs()) * 0.00391 + 1e-6;
    for k in 0..16 {
        assert!((s * f32::from(qs[k] & 0x0f) + bias - src[k]).abs() <= tol);
        assert!((s * f32::from(qs[k] >> 4) + bias - src[k + 16]).abs() <= tol);
    }
    // A flat block: scale 0, bias the value, quants 0.
    quantize_f32_to_q4(&[3.5; 32], &mut qs, &mut sc, &mut bi);
    assert_eq!((bf16_to_f32(sc[0]), bf16_to_f32(bi[0])), (0.0, 3.5));
    assert!(qs.iter().all(|&q| q == 0));
}

#[test]
fn gemv_q4_matches_reference() {
    let mut rng = Rng::new(0x40e5);
    for (rows, cols) in [
        (1usize, 32usize),
        (8, 64),
        (64, 128),
        (256, 256),
        (512, 512),
        (1024, 2048),
        (300, 640),
    ] {
        let w = rng.vec(rows * cols, -1.0, 1.0);
        let m = quantized(&w, rows, cols);
        let x = rng.vec(cols, -1.0, 1.0);
        let xsum = block_sums_of(&x, 1, cols);
        let want = gemv_ref(&q4_dequant(&m.q4.0, &m.q4.1, &m.q4.2), &x, rows, cols);
        for be in Backend::all() {
            let mut y = vec![0.0; rows];
            be.gemv_q4(&m.q4.0, &m.q4.1, &m.q4.2, &x, &xsum, &mut y, rows, cols);
            assert_close(&y, &want, 1e-3, &format!("{be:?} gemv_q4 {rows}x{cols}"));
        }
    }
}

/// The batched gemvs (three tokens per kernel call, the rest one at a time)
/// give every vector the bits the single-vector gemv gives it.
#[test]
fn gemv_batch_matches_rows() {
    let mut rng = Rng::new(0xba7c);
    for n_batch in [2usize, 3, 4, 7] {
        for (rows, cols) in [(64usize, 256usize), (512, 2048), (68, 96), (6, 64)] {
            let w = rng.vec(rows * cols, -1.0, 1.0);
            let m = quantized(&w, rows, cols);
            let wb: Vec<u16> = w.iter().map(|&v| f32_to_bf16(v)).collect();
            // Strided vectors, padding included, as the model lays them out.
            let (xs, ys) = (cols + 32, rows + 3);
            let x = rng.vec(n_batch * xs, -1.0, 1.0);
            let xsum: Vec<f32> = (0..n_batch)
                .flat_map(|t| block_sums_of(&x[t * xs..][..cols], 1, cols))
                .collect();
            for be in Backend::all() {
                let what = format!("{be:?} n_batch={n_batch} {rows}x{cols}");
                let mut yb = vec![0.0; n_batch * ys];
                let mut y = vec![0.0; rows];
                be.gemv_q8_0_batch(&m.q8.0, &m.q8.1, &x, xs, &mut yb, ys, rows, cols, n_batch);
                for t in 0..n_batch {
                    be.gemv_q8_0(&m.q8.0, &m.q8.1, &x[t * xs..][..cols], &mut y, rows, cols);
                    assert_eq!(bits(&y), bits(&yb[t * ys..][..rows]), "{what} q8_0 token {t}");
                }
                let nb = cols / 32;
                be.gemv_q4_batch(
                    &m.q4.0, &m.q4.1, &m.q4.2, &x, xs, &xsum, nb, &mut yb, ys, rows, cols, n_batch,
                );
                for t in 0..n_batch {
                    be.gemv_q4(
                        &m.q4.0,
                        &m.q4.1,
                        &m.q4.2,
                        &x[t * xs..][..cols],
                        &xsum[t * nb..][..nb],
                        &mut y,
                        rows,
                        cols,
                    );
                    assert_eq!(bits(&y), bits(&yb[t * ys..][..rows]), "{what} q4 token {t}");
                }
                be.gemv_bf16_batch(&wb, &x, xs, &mut yb, ys, rows, cols, n_batch);
                for t in 0..n_batch {
                    be.gemv_bf16(&wb, &x[t * xs..][..cols], &mut y, rows, cols);
                    assert_eq!(bits(&y), bits(&yb[t * ys..][..rows]), "{what} bf16 token {t}");
                }
            }
        }
    }
}

/// Splitting rows across threads never changes a row's arithmetic.
#[test]
fn gemv_results_do_not_depend_on_thread_count() {
    let _g = crate::pool::test_lock();
    let mut rng = Rng::new(0x7417);
    let (rows, cols, n_batch) = (1024, 512, 5);
    let w = rng.vec(rows * cols, -1.0, 1.0);
    let m = quantized(&w, rows, cols);
    let wb: Vec<u16> = w.iter().map(|&v| f32_to_bf16(v)).collect();
    let x = rng.vec(n_batch * cols, -1.0, 1.0);
    let xsum = block_sums_of(&x, n_batch, cols);
    let run = |be: Backend| {
        let mut out = vec![0.0; 6 * n_batch * rows];
        let (single, batch) = out.split_at_mut(3 * rows);
        let (s8, rest) = single.split_at_mut(rows);
        let (s4, sb) = rest.split_at_mut(rows);
        be.gemv_q8_0(&m.q8.0, &m.q8.1, &x, s8, rows, cols);
        be.gemv_q4(&m.q4.0, &m.q4.1, &m.q4.2, &x, &xsum, s4, rows, cols);
        be.gemv_bf16(&wb, &x, sb, rows, cols);
        let (b8, rest) = batch.split_at_mut(n_batch * rows);
        let (b4, bb) = rest.split_at_mut(n_batch * rows);
        be.gemv_q8_0_batch(&m.q8.0, &m.q8.1, &x, cols, b8, rows, rows, cols, n_batch);
        be.gemv_q4_batch(
            &m.q4.0,
            &m.q4.1,
            &m.q4.2,
            &x,
            cols,
            &xsum,
            cols / 32,
            b4,
            rows,
            rows,
            cols,
            n_batch,
        );
        be.gemv_bf16_batch(&wb, &x, cols, &mut bb[..n_batch * rows], rows, rows, cols, n_batch);
        (bits(&out), be.gemv_bf16_argmax(&wb, &x, rows, cols))
    };
    for be in Backend::all() {
        crate::pool::set_num_threads(1);
        let one = run(be);
        for n in [2, 3, 7, 16] {
            crate::pool::set_num_threads(n);
            assert!(run(be) == one, "{be:?}: {n} threads differ from 1");
        }
    }
    crate::pool::set_num_threads(0);
}

// ---- load-time dequantizers ----------------------------------------------------------------

/// IEEE binary32 -> binary16, round to nearest even (normals only), for
/// building synthetic blocks.
fn fp32_to_fp16(f: f32) -> u16 {
    let b = f.to_bits();
    let s = (b >> 31) & 1;
    let mut e = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let m = b & 0x7f_ffff;
    let inf = ((s << 15) | (0x1f << 10)) as u16;
    if e >= 31 {
        return inf;
    }
    if e <= 0 {
        return (s << 15) as u16;
    }
    let round = (m >> 12) & 1;
    let sticky = u32::from(m & 0xfff != 0);
    let mut mh = (m >> 13) + (round & (sticky | ((m >> 13) & 1)));
    if mh & 0x400 != 0 {
        mh = 0;
        e += 1;
        if e >= 31 {
            return inf;
        }
    }
    ((s << 15) | ((e as u32) << 10) | (mh & 0x3ff)) as u16
}

/// A Q4_K block whose super-scale and sub-block scales are 1 and mins 0, so
/// it decodes to the nibbles themselves.
fn q4k_unit_block(qs: [u8; 128]) -> [u8; 144] {
    let mut b = [0u8; 144];
    b[..2].copy_from_slice(&fp32_to_fp16(1.0).to_le_bytes());
    // Sub-blocks 0..3: scale in the low 6 bits of bytes 4..8 (mins 0 in
    // 8..12); 4..7: low 4 bits of bytes 12..16, high bits from 4..8 (0).
    b[4..8].fill(1);
    b[12..16].fill(1);
    b[16..].copy_from_slice(&qs);
    b
}

#[test]
fn dequantize_q8_0() {
    // Two blocks with distinct scales: d * q exactly.
    let mut blocks = [0u8; 68];
    blocks[..2].copy_from_slice(&fp32_to_fp16(0.25).to_le_bytes());
    blocks[34..36].copy_from_slice(&fp32_to_fp16(-2.0).to_le_bytes());
    for k in 0..32 {
        blocks[2 + k] = (k as i8 - 16) as u8;
        blocks[36 + k] = (127 - 4 * k as i32) as i8 as u8;
    }
    let mut out = [0.0f32; 64];
    assert!(dequantize_to_f32(8, &blocks, &mut out));
    for k in 0..32 {
        assert_eq!(out[k], 0.25 * (k as f32 - 16.0));
        assert_eq!(out[32 + k], -2.0 * (127 - 4 * k as i32) as f32);
    }
    assert_eq!(
        (quant_block_elems(8), quant_block_elems(14), quant_block_elems(30)),
        (32, 256, 0)
    );
    assert_eq!((quant_block_elems(0), quant_block_elems(9)), (0, 0));
}

#[test]
fn dequantize_legacy() {
    // One block of each classic format, through the type dispatcher.
    let (d, m) = (0.5f32, -1.25f32);
    for (ty, has_min, has_high) in [
        (2u32, false, false),
        (3, true, false),
        (6, false, true),
        (7, true, true),
    ] {
        let q: Vec<i32> = (0..32)
            .map(|j| if has_high { (j * 11 + 5) % 32 } else { (j * 7 + 3) % 16 })
            .collect();
        let mut block = Vec::new();
        block.extend_from_slice(&fp32_to_fp16(d).to_le_bytes());
        if has_min {
            block.extend_from_slice(&fp32_to_fp16(m).to_le_bytes());
        }
        if has_high {
            let qh = (0..32).fold(0u32, |h, j| h | (((q[j] >> 4) & 1) as u32) << j);
            block.extend_from_slice(&qh.to_le_bytes());
        }
        block.extend((0..16).map(|j| ((q[j] & 0x0f) | ((q[j + 16] & 0x0f) << 4)) as u8));
        let mut out = [0.0f32; 32];
        assert_eq!(quant_block_elems(ty), 32);
        assert!(dequantize_to_f32(ty, &block, &mut out));
        let centre = if has_min {
            0
        } else if has_high {
            16
        } else {
            8
        };
        for j in 0..32 {
            assert_eq!(
                out[j],
                (q[j] - centre) as f32 * d + if has_min { m } else { 0.0 },
                "type {ty} [{j}]"
            );
        }
    }
}

#[test]
fn dequantize_q4_k() {
    // Low nibbles 1, high nibbles 2: each 64-run is 32 ones then 32 twos.
    let block = q4k_unit_block([0x21; 128]);
    let mut out = [0.0f32; 256];
    assert!(dequantize_to_f32(12, &block, &mut out));
    for (j, run) in out.as_chunks::<64>().0.iter().enumerate() {
        assert!(
            run[..32].iter().all(|&v| v == 1.0) && run[32..].iter().all(|&v| v == 2.0),
            "run {j}"
        );
    }
}

#[test]
fn dequantize_q5_k() {
    // Q4_K's scale layout plus qh, a fifth bit per element.
    let mut b = [0u8; 176];
    b[..2].copy_from_slice(&fp32_to_fp16(1.0).to_le_bytes());
    b[4..8].fill(1);
    b[12..16].fill(1);
    b[48..].fill(0x21);
    let mut out = [0.0f32; 256];
    for (qh, lo, hi) in [(0x00u8, 1.0f32, 2.0f32), (0xff, 17.0, 18.0)] {
        b[16..48].fill(qh);
        assert!(dequantize_to_f32(13, &b, &mut out));
        for run in out.as_chunks::<64>().0 {
            assert!(
                run[..32].iter().all(|&v| v == lo) && run[32..].iter().all(|&v| v == hi),
                "qh {qh:#x}"
            );
        }
    }
}

#[test]
fn fp16_subnormal_dequant() {
    // Regression: a subnormal fp16 super-scale (K-quant scales often are)
    // once decoded 2x too small. d = 2^-24, sub-block 0 scale 1, q = 1.
    let mut b = [0u8; 144];
    b[..2].copy_from_slice(&1u16.to_le_bytes());
    b[4] = 1;
    b[16..].fill(0x11);
    let mut out = [0.0f32; 256];
    assert!(dequantize_to_f32(12, &b, &mut out));
    assert_eq!(out[0], 2f32.powi(-24));
}

#[test]
fn dequantize_q6_k() {
    // d = 1, scales 1, ql = 0x10 (low 0, high 1), qh = 0:
    // q1 = q2 = -32, q3 = q4 = -31.
    let mut b = [0u8; 210];
    b[..128].fill(0x10);
    b[192..208].fill(1);
    b[208..].copy_from_slice(&fp32_to_fp16(1.0).to_le_bytes());
    let mut out = [0.0f32; 256];
    assert!(dequantize_to_f32(14, &b, &mut out));
    for half in out.as_chunks::<128>().0 {
        assert!(half[..64].iter().all(|&v| v == -32.0) && half[64..].iter().all(|&v| v == -31.0));
    }
}

#[test]
fn dequantize_float_types_and_rejects() {
    let vals = [1.5f32, -2.25, 0.0, 3e-3, -7.0];
    let f32_bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    let f16_bytes: Vec<u8> = vals.iter().flat_map(|&v| fp32_to_fp16(v).to_le_bytes()).collect();
    let bf16_bytes: Vec<u8> = vals.iter().flat_map(|&v| f32_to_bf16(v).to_le_bytes()).collect();
    let mut out = [0.0f32; 5];
    assert!(dequantize_to_f32(0, &f32_bytes, &mut out) && out == vals);
    assert!(dequantize_to_f32(1, &f16_bytes, &mut out));
    assert!(out.iter().zip(&vals).all(|(&o, &v)| (o - v).abs() <= v.abs() * 1e-3));
    assert!(dequantize_to_f32(30, &bf16_bytes, &mut out));
    assert!(out.iter().zip(&vals).all(|(&o, &v)| o == bf16_to_f32(f32_to_bf16(v))));
    // Unknown types, partial blocks and short input leave dst untouched.
    let mut dst = [9.0f32; 32];
    assert!(!dequantize_to_f32(9, &[0; 64], &mut dst));
    assert!(!dequantize_to_f32(10, &[0; 128], &mut dst));
    assert!(!dequantize_to_f32(8, &[0; 33], &mut dst));
    assert!(!dequantize_to_f32(8, &[0; 68], &mut [0.0; 33]));
    assert!(dst.iter().all(|&v| v == 9.0));
}

// ---- RoPE, embedding rows, routing ------------------------------------------------------------

#[test]
fn rope_matches_reference() {
    let mut rng = Rng::new(0x5051);
    // pos = 0 is the identity.
    let x0 = rng.vec(8 * 64, -1.0, 1.0);
    let mut x = x0.clone();
    rope(&mut x, 8, 64, 64, 0.0, 10000.0, RopeMode::Neox, None);
    assert_close(&x, &x0, 1e-6, "rope pos 0");
    // Against a direct evaluation, both pairings, n_rot < head_dim (the
    // trailing lanes pass through), with and without frequency factors
    // (1e30 switches a pair off).
    let (h, d, r, pos, base) = (4usize, 64usize, 32usize, 17.0f32, 1e6f32);
    let half = r / 2;
    let ff: Vec<f32> = (0..half)
        .map(|i| if i == 3 { 1e30 } else { 1.0 + i as f32 * 0.25 })
        .collect();
    for mode in [RopeMode::Neox, RopeMode::Norm] {
        for factors in [None, Some(&ff[..])] {
            let x0 = rng.vec(h * d, -1.0, 1.0);
            let mut want = x0.clone();
            for xh in want.chunks_exact_mut(d) {
                for i in 0..half {
                    let inv = base.powf(-((2 * i) as f32) / r as f32);
                    let theta = pos * inv / factors.map_or(1.0, |f| f[i]);
                    let (c, s) = (theta.cos(), theta.sin());
                    let (p, q) = if mode == RopeMode::Neox {
                        (i, i + half)
                    } else {
                        (2 * i, 2 * i + 1)
                    };
                    let (a, b) = (xh[p], xh[q]);
                    xh[p] = c * a - s * b;
                    xh[q] = s * a + c * b;
                }
            }
            let mut x = x0.clone();
            rope(&mut x, h, d, r, pos, base, mode, factors);
            assert_close(&x, &want, 1e-5, &format!("rope {mode:?} {}", factors.is_some()));
            for xh in x.chunks_exact(d).zip(x0.chunks_exact(d)) {
                assert_eq!(bits(&xh.0[r..]), bits(&xh.1[r..]), "rope pass-through lanes");
            }
        }
    }
    // Rotation preserves each pair's magnitude.
    let x0 = rng.vec(2 * 16, -1.0, 1.0);
    let mut x = x0.clone();
    rope(&mut x, 2, 16, 16, 42.0, 10000.0, RopeMode::Neox, None);
    for (a, b) in x0.as_chunks::<16>().0.iter().zip(x.as_chunks::<16>().0) {
        for i in 0..8 {
            let (m0, m1) = (a[i] * a[i] + a[i + 8] * a[i + 8], b[i] * b[i] + b[i + 8] * b[i + 8]);
            assert!((m0 - m1).abs() <= 1e-4 * m0.max(1e-6));
        }
    }
}

#[test]
fn embed_rows_match_dequantized_tables() {
    let mut rng = Rng::new(0xe3b0);
    // BF16, with an n_embd off the 8-wide path.
    let (n_embd, n_vocab) = (17, 5);
    let table = rng.bf16s(n_vocab * n_embd, -2.0, 2.0);
    let mut y = vec![0.0; n_embd];
    embed_row_bf16(&mut y, &table, 3, 1.5);
    for (i, &v) in y.iter().enumerate() {
        assert_eq!(v, bf16_to_f32(table[3 * n_embd + i]) * 1.5);
    }
    // Q8_0 and Q4: the scaled dequantized row, within quantization error of the source.
    let (n_embd, n_vocab, tok, scale) = (64, 7, 5, 0.75f32);
    let src = rng.vec(n_vocab * n_embd, -2.0, 2.0);
    let (mut qs, mut sc) = (vec![0i8; src.len()], vec![0u16; src.len() / 32]);
    quantize_f32_to_q8_0(&src, &mut qs, &mut sc);
    let mut y = vec![0.0; n_embd];
    embed_row_q8_0(&mut y, &qs, &sc, tok, scale);
    let row = tok * n_embd;
    for (i, &v) in y.iter().enumerate() {
        assert!(close(
            v,
            f32::from(qs[row + i]) * bf16_to_f32(sc[(row + i) / 32]) * scale,
            1e-5
        ));
        assert!((v - src[row + i] * scale).abs() <= 0.05);
    }
    let (mut q4, mut s4, mut b4) = (
        vec![0u8; src.len() / 2],
        vec![0u16; src.len() / 32],
        vec![0u16; src.len() / 32],
    );
    quantize_f32_to_q4(&src, &mut q4, &mut s4, &mut b4);
    embed_row_q4(&mut y, &q4, &s4, &b4, tok, scale);
    let want = q4_dequant(&q4[row / 2..][..n_embd / 2], &s4[row / 32..][..2], &b4[row / 32..][..2]);
    for (i, (&v, &w)) in y.iter().zip(&want).enumerate() {
        assert!(close(v, w * scale, 1e-5));
        // 4 bits over a [-2, 2] spread: at most half a step, ~0.13.
        assert!((v - src[row + i] * scale).abs() <= 0.2);
    }
}

#[test]
fn moe_top_k_breaks_ties_low() {
    let p = [0.10f32, 0.30, 0.05, 0.30, 0.20, 0.05];
    let (mut idx, mut w) = ([0i32; 3], [0.0f32; 3]);
    moe_top_k(&p, 3, &mut idx, &mut w);
    // 0.30 twice: the lower index first, then 0.20.
    assert_eq!(idx, [1, 3, 4]);
    assert!(close(w[0], 0.375, 1e-6) && close(w[1], 0.375, 1e-6) && close(w[2], 0.25, 1e-6));
    moe_top_k(&p, 6, &mut [0; 6], &mut [0.0; 6]);
}

// ---- Gated DeltaNet ------------------------------------------------------------------------------

#[test]
fn conv1d_silu_matches_reference() {
    let mut rng = Rng::new(13);
    for be in Backend::all() {
        // 21 channels leave a scalar tail.
        for ch in [21usize, 16, 3] {
            let (k, n_tok) = (4, 3);
            let x = rng.normals((k - 1 + n_tok) * ch, 1.0);
            let w = rng.normals(k * ch, 1.0);
            let mut y = vec![0.0; n_tok * ch];
            be.conv1d_silu(&mut y, &x, &w, n_tok, ch, k);
            for t in 0..n_tok {
                for c in 0..ch {
                    let acc: f64 = (0..k)
                        .map(|j| f64::from(w[j * ch + c]) * f64::from(x[(t + j) * ch + c]))
                        .sum();
                    let want = (acc / (1.0 + (-acc).exp())) as f32;
                    assert!(close(y[t * ch + c], want, 1e-5), "{be:?} conv1d ch={ch} [{t}][{c}]");
                }
            }
        }
    }
}

#[test]
fn delta_rule_matches_textbook_form() {
    // S = decay S;  S += beta (v - S k) k^T;  o = S q, over several steps.
    let (d_k, d_v) = (43usize, 5usize); // 43 = 2 * 16 + 8 + 3: every loop of the kernel
    for be in Backend::all() {
        let mut rng = Rng::new(14);
        let (mut s, mut r) = (vec![0.0f32; d_v * d_k], vec![0.0f32; d_v * d_k]);
        let mut o = vec![0.0; d_v];
        for _ in 0..6 {
            let (q, v) = (rng.normals(d_k, 1.0), rng.normals(d_v, 1.0));
            let mut k = rng.normals(d_k, 1.0);
            let kk = dot_f64(&k, &k).sqrt();
            k.iter_mut().for_each(|x| *x = (f64::from(*x) / kk) as f32);
            let (decay, beta) = (0.5 + 0.5 * rng.unit(), rng.unit());
            be.delta_rule(&mut s, &q, &k, &v, decay, beta, &mut o, d_k, d_v);
            for j in 0..d_v {
                let row = &mut r[j * d_k..][..d_k];
                row.iter_mut().for_each(|x| *x *= decay);
                let delta = f64::from(beta) * (f64::from(v[j]) - dot_f64(row, &k));
                row.iter_mut()
                    .zip(&k)
                    .for_each(|(x, &k)| *x = (f64::from(*x) + delta * f64::from(k)) as f32);
                assert!(close(o[j], dot_f64(row, &q) as f32, 1e-4), "{be:?} o[{j}]");
            }
            assert_close(&s, &r, 1e-4, &format!("{be:?} state"));
        }
    }
}

// ---- attention ----------------------------------------------------------------------------------------

/// Scalar softmax attention of `group` heads over n_keys K/V rows.
fn ref_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    kv_stride: usize,
    n_keys: usize,
    hd: usize,
    group: usize,
    scale: f32,
) -> Vec<f32> {
    let mut out = vec![0.0; group * hd];
    for h in 0..group {
        let s: Vec<f64> = (0..n_keys)
            .map(|t| dot_f64(&q[h * hd..][..hd], &k[t * kv_stride..][..hd]) * f64::from(scale))
            .collect();
        let m = s.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let e: Vec<f64> = s.iter().map(|&s| (s - m).exp()).collect();
        let sum: f64 = e.iter().sum();
        for i in 0..hd {
            let o: f64 = (0..n_keys).map(|t| e[t] * f64::from(v[t * kv_stride + i])).sum();
            out[h * hd + i] = (o / sum) as f32;
        }
    }
    out
}

/// Blocks of keys merged in order give softmax attention over them all, with
/// F32 and BF16 K/V, head sizes on and off the 8-wide path, and head counts
/// that use the 4-head score kernel and its remainders.
#[test]
fn attn_blocks_match_softmax_attention() {
    let mut rng = Rng::new(7);
    for bf16 in [false, true] {
        for (hd, group) in [(64usize, 3usize), (36, 3), (64, 5), (40, 4)] {
            let (n_kv, n_keys) = (2, 600);
            let kv_stride = n_kv * hd;
            let scale = 1.0 / (hd as f32).sqrt();
            let q = rng.normals(group * hd, 1.0);
            let mut kf = rng.normals(n_keys * kv_stride, 2.0);
            let mut vf = rng.normals(n_keys * kv_stride, 1.0);
            let kb: Vec<u16> = kf.iter().map(|&x| f32_to_bf16(x)).collect();
            let vb: Vec<u16> = vf.iter().map(|&x| f32_to_bf16(x)).collect();
            if bf16 {
                kf = kb.iter().map(|&x| bf16_to_f32(x)).collect();
                vf = vb.iter().map(|&x| bf16_to_f32(x)).collect();
            }
            // KV head 1, in blocks of 100, 256 and 244 keys.
            let rows = |k0: usize| {
                let off = k0 * kv_stride + hd;
                if bf16 {
                    (KvRows::Bf16(&kb[off..]), KvRows::Bf16(&vb[off..]))
                } else {
                    (KvRows::F32(&kf[off..]), KvRows::F32(&vf[off..]))
                }
            };
            let want = ref_attention(&q, &kf[hd..], &vf[hd..], kv_stride, n_keys, hd, group, scale);
            let pw = group * attn_part(hd);
            let blocks = [(0usize, 100usize), (100, 256), (356, 244)];
            for be in Backend::all() {
                let mut scores = vec![0.0; group * 256];
                let (mut acc, mut part) = (vec![0.0; pw], vec![0.0; pw]);
                for (i, &(k0, n)) in blocks.iter().enumerate() {
                    let (k, v) = rows(k0);
                    let dst = if i == 0 { &mut acc } else { &mut part };
                    be.attn_block(&q, hd, k, v, kv_stride, n, hd, group, scale, &mut scores, dst);
                    if i > 0 {
                        be.attn_merge(&mut acc, &part, hd, group);
                    }
                }
                let mut out = vec![0.0; group * hd];
                attn_finish(&mut out, hd, &acc, hd, group);
                for (i, (&o, &w)) in out.iter().zip(&want).enumerate() {
                    assert!(
                        (o - w).abs() <= 1e-4 * (1.0 + w.abs()),
                        "{be:?} bf16={bf16} hd={hd} [{i}]: {o} vs {w}"
                    );
                }
                // The same blocks computed on the pool, each into its own
                // partial, then merged in order: the same bits.
                let parts: Vec<std::sync::Mutex<Vec<f32>>> =
                    blocks.iter().map(|_| std::sync::Mutex::new(vec![0.0; pw])).collect();
                crate::pool::parallel_for(blocks.len() as u32, |b0, b1| {
                    for b in b0 as usize..b1 as usize {
                        let (k0, n) = blocks[b];
                        let (k, v) = rows(k0);
                        let mut sc = vec![0.0; group * 256];
                        be.attn_block(
                            &q,
                            hd,
                            k,
                            v,
                            kv_stride,
                            n,
                            hd,
                            group,
                            scale,
                            &mut sc,
                            &mut parts[b].lock().unwrap(),
                        );
                    }
                });
                let mut acc2 = parts[0].lock().unwrap().clone();
                for p in &parts[1..] {
                    be.attn_merge(&mut acc2, &p.lock().unwrap(), hd, group);
                }
                assert_eq!(bits(&acc2), bits(&acc), "{be:?} parallel blocks");
            }
        }
    }
}

/// AVX2 tails go through the lane functions with a fused multiply-add: a
/// tail element gets the bits a vector lane gets for the same input.
#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_tails_match_vector_lanes() {
    let Some(k) = avx2::Avx2::detect() else { return };
    let be = Backend::Avx2(k);
    let mut rng = Rng::new(77);
    let lanes = rng.vec(8, -6.0, 6.0);
    let x: Vec<f32> = lanes.iter().chain(&lanes[..7]).copied().collect();
    let u: Vec<f32> = x.iter().map(|v| v * 0.5 + 0.25).collect();
    let tail_matches = |y: &[f32], what: &str| {
        for i in 0..7 {
            assert_eq!(
                y[8 + i].to_bits(),
                y[i].to_bits(),
                "{what}: tail [{}] vs lane [{i}]",
                8 + i
            );
        }
    };
    let mut y = x.clone();
    be.gelu_inplace(&mut y);
    tail_matches(&y, "gelu");
    let mut y = x.clone();
    be.gelu_mul_inplace(&mut y, &u);
    tail_matches(&y, "gelu_mul");
    let mut y = x.clone();
    be.swiglu_inplace(&mut y, &u);
    tail_matches(&y, "swiglu");
    let mut y = u.clone();
    be.mul_sigmoid(&mut y, &x);
    tail_matches(&y, "mul_sigmoid");
    let mut y = x.clone();
    be.softmax_inplace(&mut y);
    tail_matches(&y, "softmax");
    // conv1d: channel c + 8 repeats channel c in every row.
    let (ch, kk) = (15, 3);
    let xs: Vec<f32> = (0..kk).flat_map(|_| x.iter().map(|v| v * 0.3)).collect();
    let ws: Vec<f32> = (0..kk)
        .flat_map(|j| u.iter().map(move |v| v - j as f32 * 0.1))
        .collect();
    let mut y = vec![0.0; ch];
    be.conv1d_silu(&mut y, &xs, &ws, 1, ch, kk);
    tail_matches(&y, "conv1d_silu");
}

/// The portable exp is the AVX2 polynomial: ~3e-7 relative over the range.
#[test]
fn exp_lane_is_accurate() {
    let mut worst = 0.0f64;
    for i in 0..=20000 {
        let x = -87.0 + 175.0 * i as f32 / 20000.0;
        let got = f64::from(scalar::exp_lane(x, f32::mul_add));
        worst = worst.max((got / f64::from(x).exp() - 1.0).abs());
    }
    assert!(worst < 1e-6, "exp worst relative error {worst}");
}

// ---- benchmark -------------------------------------------------------------------------------------------

fn random_bytes(rng: &mut Rng, n: usize) -> Vec<u8> {
    let mut v = vec![0u8; n];
    for c in v.chunks_mut(8) {
        c.copy_from_slice(&rng.next_u64().to_le_bytes()[..c.len()]);
    }
    v
}

/// Seconds of the fastest of `reps` runs, after one warm-up.
fn best_secs(reps: usize, mut f: impl FnMut()) -> f64 {
    f();
    (0..reps)
        .map(|_| {
            let t = std::time::Instant::now();
            f();
            t.elapsed().as_secs_f64()
        })
        .fold(f64::INFINITY, f64::min)
}

/// Weight-streaming rates of the Q8_0 and Q4 gemvs, one thread and the
/// pool. Each pass cycles through enough copies of the matrix (>= 256 MB)
/// that the weights come from DRAM, as in decode, not from the 32 MB L3
/// ("hot" repeats one copy). Also a 16-token batch, prefill's unit.
#[test]
#[ignore = "benchmark: cargo test --release -p nncpu -- --ignored --nocapture bench_gemv"]
fn bench_gemv() {
    let _g = crate::pool::test_lock();
    let mut rng = Rng::new(1);
    println!("\nnncpu gemv benchmark, backend {:?}", Backend::current());
    {
        // Calibration: a 512 MB stream with almost no work per byte.
        crate::pool::set_num_threads(1);
        let big: Vec<f32> = random_bytes(&mut rng, 512 << 20)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from(b[0]))
            .collect();
        let t = best_secs(3, || {
            std::hint::black_box(dot(&big, &big));
        });
        println!(
            "load-only roofline, 1 thread: {:.1} GB/s",
            (big.len() * 4) as f64 / t / 1e9
        );
    }
    println!(
        "{:<12} {:<6} {:>7} {:>9} {:>9} {:>10}",
        "shape", "format", "threads", "GB/s", "hot GB/s", "us/gemv"
    );
    for (rows, cols) in [(8192usize, 2048usize), (262144, 640)] {
        let (n, nb) = (rows * cols, rows * cols / 32);
        let scales = |rng: &mut Rng| -> Vec<u16> { (0..nb).map(|_| f32_to_bf16(rng.uniform(1e-3, 2e-3))).collect() };
        let q8_bytes = n + 2 * nb;
        let q4_bytes = n / 2 + 4 * nb;
        let copies = |bytes: usize| ((256 << 20) / bytes + 1).max(2);
        let q8: Vec<(Vec<i8>, Vec<u16>)> = (0..copies(q8_bytes))
            .map(|_| {
                (
                    random_bytes(&mut rng, n).into_iter().map(|b| b as i8).collect(),
                    scales(&mut rng),
                )
            })
            .collect();
        let q4: Vec<(Vec<u8>, Vec<u16>, Vec<u16>)> = (0..copies(q4_bytes))
            .map(|_| (random_bytes(&mut rng, n / 2), scales(&mut rng), scales(&mut rng)))
            .collect();
        // BF16 weights of magnitude ~2^-7: random sign and mantissa, no NaN or subnormals.
        let bf16_bytes = 2 * n;
        let bf: Vec<Vec<u16>> = (0..copies(bf16_bytes))
            .map(|_| {
                random_bytes(&mut rng, 2 * n)
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| (u16::from_le_bytes([b[0], b[1]]) & 0x80ff) | 0x3c00)
                    .collect()
            })
            .collect();
        let x = rng.vec(cols, -1.0, 1.0);
        let xsum = block_sums_of(&x, 1, cols);
        let mut y = vec![0.0; rows];
        for threads in [1, 0] {
            crate::pool::set_num_threads(threads);
            let nt = crate::pool::num_threads();
            let t = best_secs(3, || {
                q8.iter().for_each(|(q, s)| gemv_q8_0(q, s, &x, &mut y, rows, cols))
            }) / q8.len() as f64;
            let hot = best_secs(10, || gemv_q8_0(&q8[0].0, &q8[0].1, &x, &mut y, rows, cols));
            let gbs = |secs: f64, bytes: usize| bytes as f64 / secs / 1e9;
            println!(
                "{:<12} {:<6} {:>7} {:>9.1} {:>9.1} {:>10.0}",
                format!("{rows}x{cols}"),
                "Q8_0",
                nt,
                gbs(t, q8_bytes),
                gbs(hot, q8_bytes),
                t * 1e6
            );
            let t = best_secs(3, || {
                q4.iter()
                    .for_each(|(q, s, b)| gemv_q4(q, s, b, &x, &xsum, &mut y, rows, cols))
            }) / q4.len() as f64;
            let (q, s, b) = &q4[0];
            let hot = best_secs(10, || gemv_q4(q, s, b, &x, &xsum, &mut y, rows, cols));
            println!(
                "{:<12} {:<6} {:>7} {:>9.1} {:>9.1} {:>10.0}",
                format!("{rows}x{cols}"),
                "Q4",
                nt,
                gbs(t, q4_bytes),
                gbs(hot, q4_bytes),
                t * 1e6
            );
            let t = best_secs(3, || bf.iter().for_each(|w| gemv_bf16(w, &x, &mut y, rows, cols))) / bf.len() as f64;
            let hot = best_secs(10, || gemv_bf16(&bf[0], &x, &mut y, rows, cols));
            println!(
                "{:<12} {:<6} {:>7} {:>9.1} {:>9.1} {:>10.0}",
                format!("{rows}x{cols}"),
                "BF16",
                nt,
                gbs(t, bf16_bytes),
                gbs(hot, bf16_bytes),
                t * 1e6
            );
            let t = best_secs(3, || {
                bf.iter().for_each(|w| {
                    std::hint::black_box(gemv_bf16_argmax(w, &x, rows, cols));
                })
            }) / bf.len() as f64;
            println!(
                "{:<12} {:<6} {:>7} {:>9.1} {:>9} {:>10.0}",
                format!("{rows}x{cols}"),
                "argmax",
                nt,
                gbs(t, bf16_bytes),
                "",
                t * 1e6
            );
        }
    }
    // Prefill: a 16-token batch against 8192x2048, three tokens per kernel call.
    let (rows, cols, n_batch) = (8192usize, 2048usize, 16usize);
    let w = rng.vec(rows * cols, -1.0, 1.0);
    let m = quantized(&w, rows, cols);
    let x = rng.vec(n_batch * cols, -1.0, 1.0);
    let xsum = block_sums_of(&x, n_batch, cols);
    let mut y = vec![0.0; n_batch * rows];
    for threads in [1, 0] {
        crate::pool::set_num_threads(threads);
        let nt = crate::pool::num_threads();
        let t8 = best_secs(5, || {
            gemv_q8_0_batch(&m.q8.0, &m.q8.1, &x, cols, &mut y, rows, rows, cols, n_batch)
        });
        let t4 = best_secs(5, || {
            gemv_q4_batch(
                &m.q4.0,
                &m.q4.1,
                &m.q4.2,
                &x,
                cols,
                &xsum,
                cols / 32,
                &mut y,
                rows,
                rows,
                cols,
                n_batch,
            )
        });
        let gflops = |t: f64| 2.0 * (rows * cols * n_batch) as f64 / t / 1e9;
        println!(
            "batch {n_batch} {rows}x{cols}, {nt} threads: Q8_0 {:.0} us ({:.1} GFLOP/s), Q4 {:.0} us ({:.1} GFLOP/s)",
            t8 * 1e6,
            gflops(t8),
            t4 * 1e6,
            gflops(t4)
        );
    }
    // Attention: one 256-key block of BF16 K/V for 4 query heads sharing a
    // KV head (Gemma 3 1B's GQA), head_dim 256, from L2.
    let (hd, group, n_keys) = (256usize, 4usize, 256usize);
    let q = rng.vec(group * hd, -1.0, 1.0);
    let (k, v) = (rng.bf16s(n_keys * hd, -1.0, 1.0), rng.bf16s(n_keys * hd, -1.0, 1.0));
    let (mut scores, mut part) = (vec![0.0; group * n_keys], vec![0.0; group * attn_part(hd)]);
    let t = best_secs(500, || {
        attn_block(
            &q,
            hd,
            KvRows::Bf16(&k),
            KvRows::Bf16(&v),
            hd,
            n_keys,
            hd,
            group,
            0.0625,
            &mut scores,
            &mut part,
        )
    });
    println!(
        "attn_block {group} heads x {n_keys} keys, head_dim {hd}, BF16 K/V: {:.1} us ({:.1} GB/s of K/V)",
        t * 1e6,
        (2 * n_keys * hd * 2) as f64 / t / 1e9
    );
    crate::pool::set_num_threads(0);
}
