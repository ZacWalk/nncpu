//! CPU feature detection.
//!
//! The fast kernels need x86-64 with AVX2 and FMA (and an OS that saves the
//! YMM registers, which `is_x86_feature_detected!` checks). Everything else
//! runs portable scalar kernels: correct on any CPU, but much slower.

use std::sync::OnceLock;

/// The CPU features nncpu's kernels can use.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CpuFeatures {
    pub avx2: bool,
    pub fma: bool,
    pub f16c: bool,
}

/// The features of this CPU, detected once.
pub fn cpu_features() -> CpuFeatures {
    static FEATURES: OnceLock<CpuFeatures> = OnceLock::new();
    *FEATURES.get_or_init(detect)
}

#[cfg(target_arch = "x86_64")]
fn detect() -> CpuFeatures {
    CpuFeatures {
        avx2: std::arch::is_x86_feature_detected!("avx2"),
        fma: std::arch::is_x86_feature_detected!("fma"),
        f16c: std::arch::is_x86_feature_detected!("f16c"),
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn detect() -> CpuFeatures {
    CpuFeatures::default()
}

/// True when this CPU and OS can run nncpu's AVX2/FMA kernels: x86-64 with
/// AVX2 and FMA, with the OS saving YMM state. On other CPUs nncpu still
/// works, through portable scalar kernels that are many times slower.
pub fn cpu_supported() -> bool {
    let f = cpu_features();
    f.avx2 && f.fma
}

/// Whether the kernels take their AVX2/FMA paths: [`cpu_supported`], unless
/// the `NNCPU_NO_SIMD` environment variable is set (a debugging aid that
/// forces the scalar kernels).
#[inline]
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))] // only the x86 kernel dispatch asks
pub(crate) fn simd() -> bool {
    static SIMD: OnceLock<bool> = OnceLock::new();
    *SIMD.get_or_init(|| cpu_supported() && std::env::var_os("NNCPU_NO_SIMD").is_none())
}
