//! Quantization into nncpu's split layouts, and the load-time decoders of
//! ggml's block formats.
//!
//! Q8_0 is absmax per 32-element block (scale = max|w| / 127); Q4 is
//! asymmetric (scale = (max - min) / 15, bias = min), with byte i of a block
//! holding element i in its low nibble and i + 16 in its high one. Scales and
//! biases are rounded to BF16 before the values are quantized against them,
//! so the encoder targets exactly the numbers the kernels multiply by.

use crate::gguf::ggml_block;
use crate::runtime::{bf16_to_f32, f16_to_f32, f32_to_bf16};

/// The BF16 value nearest `v`, as an f32.
fn bf16_round(v: f32) -> f32 {
    bf16_to_f32(f32_to_bf16(v))
}

/// `lrintf`: round to nearest, ties to even, then to i32 (saturating; NaN
/// gives 0).
pub(super) fn rint(v: f32) -> i32 {
    // Adding 1.5 * 2^23 to |v| < 2^22 leaves no fraction bits, so the
    // addition does the rounding.
    const MAGIC: f32 = 12_582_912.0;
    if v.abs() < 4_194_304.0 {
        ((v + MAGIC) - MAGIC) as i32
    } else {
        v.round_ties_even() as i32
    }
}

/// Quantize `src` (a whole number of 32-blocks) into Q8_0: `qs[..n]` and
/// `scales[..n / 32]`. A flat walk gives the row-major block order of the
/// split layout for any matrix with cols % 32 == 0, so a matrix can be
/// quantized in chunks.
pub(crate) fn quantize_f32_to_q8_0(src: &[f32], qs: &mut [i8], scales: &mut [u16]) {
    let n = src.len();
    assert!(
        n.is_multiple_of(32),
        "quantize_f32_to_q8_0: {n} values are not whole 32-blocks"
    );
    let blocks = src.as_chunks::<32>().0.iter().zip(qs[..n].as_chunks_mut::<32>().0);
    for ((b, q), s) in blocks.zip(&mut scales[..n / 32]) {
        let mut amax = 0.0f32;
        for &v in b {
            if v.abs() > amax {
                amax = v.abs();
            }
        }
        // Times the reciprocal rather than divided: the C++ library is
        // built with fast-math, which compiles its `amax / 127.0f` this way,
        // and a quotient that lands exactly on a BF16 rounding tie (common
        // for weights dequantized from a quantized file) rounds differently.
        let scale = bf16_round(amax * (1.0 / 127.0));
        *s = f32_to_bf16(scale);
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        for (q, &v) in q.iter_mut().zip(b) {
            *q = rint(v * inv).clamp(-127, 127) as i8;
        }
    }
}

/// Quantize `src` (a whole number of 32-blocks) into the Q4 split layout:
/// `qs[..n / 2]` packed nibbles, `scales[..n / 32]` and `biases[..n / 32]`,
/// with w ~= scale * q + bias for q in [0, 15].
pub(crate) fn quantize_f32_to_q4(src: &[f32], qs: &mut [u8], scales: &mut [u16], biases: &mut [u16]) {
    let n = src.len();
    assert!(
        n.is_multiple_of(32),
        "quantize_f32_to_q4: {n} values are not whole 32-blocks"
    );
    let blocks = src.as_chunks::<32>().0.iter().zip(qs[..n / 2].as_chunks_mut::<16>().0);
    for ((b, q), (s, bi)) in blocks.zip(scales[..n / 32].iter_mut().zip(&mut biases[..n / 32])) {
        let (mut lo, mut hi) = (b[0], b[0]);
        for &v in &b[1..] {
            if v < lo {
                lo = v;
            }
            if v > hi {
                hi = v;
            }
        }
        // Times the reciprocal, as the C++ library's fast-math build computes
        // `(hi - lo) / 15.0f` (see quantize_f32_to_q8_0).
        let scale = bf16_round((hi - lo) * (1.0 / 15.0));
        let bias = bf16_round(lo);
        *s = f32_to_bf16(scale);
        *bi = f32_to_bf16(bias);
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        for k in 0..16 {
            let q0 = rint((b[k] - bias) * inv).clamp(0, 15);
            let q1 = rint((b[k + 16] - bias) * inv).clamp(0, 15);
            q[k] = (q0 | (q1 << 4)) as u8;
        }
    }
}

/// Elements per storage block of a ggml type code: 0 for types this build
/// cannot decode, and for the plain float types (which have no blocks).
pub(crate) fn quant_block_elems(ggml_type: u32) -> u32 {
    match ggml_type {
        2 | 3 | 6 | 7 | 8 => 32,
        12..=14 => 256,
        _ => 0,
    }
}

/// Decode `dst.len()` values from `blocks`, by ggml type code: F32 (0), F16
/// (1), BF16 (30), Q4_0 (2), Q4_1 (3), Q5_0 (6), Q5_1 (7), Q8_0 (8), Q4_K
/// (12), Q5_K (13), Q6_K (14). Returns false, leaving `dst` untouched, for
/// other types, or when `dst` is not a whole number of blocks or `blocks`
/// holds fewer than it needs.
pub(crate) fn dequantize_to_f32(ggml_type: u32, blocks: &[u8], dst: &mut [f32]) -> bool {
    if !matches!(ggml_type, 0..=3 | 6..=8 | 12..=14 | 30) {
        return false;
    }
    let Some((bytes, per)) = ggml_block(ggml_type) else {
        return false;
    };
    let (bytes, per) = (bytes as usize, per as usize);
    let n = dst.len();
    if !n.is_multiple_of(per) || blocks.len() / bytes < n / per {
        return false;
    }
    let src = &blocks[..n / per * bytes];
    let u16_at = |b: &[u8], i: usize| u16::from_le_bytes([b[i], b[i + 1]]);
    match ggml_type {
        0 => {
            for (d, b) in dst.iter_mut().zip(src.as_chunks::<4>().0) {
                *d = f32::from_le_bytes(*b);
            }
        }
        1 => {
            for (d, b) in dst.iter_mut().zip(src.as_chunks::<2>().0) {
                *d = f16_to_f32(u16::from_le_bytes(*b));
            }
        }
        30 => {
            for (d, b) in dst.iter_mut().zip(src.as_chunks::<2>().0) {
                *d = bf16_to_f32(u16::from_le_bytes(*b));
            }
        }
        8 => {
            // block_q8_0 = { fp16 d; int8 qs[32] }.
            for (b, y) in src.as_chunks::<34>().0.iter().zip(dst.as_chunks_mut::<32>().0) {
                let d = f16_to_f32(u16_at(b, 0));
                for (y, &q) in y.iter_mut().zip(&b[2..]) {
                    *y = d * f32::from(q as i8);
                }
            }
        }
        2 => dequantize_legacy::<false, false>(src, dst),
        3 => dequantize_legacy::<true, false>(src, dst),
        6 => dequantize_legacy::<false, true>(src, dst),
        7 => dequantize_legacy::<true, true>(src, dst),
        12 => dequantize_q4_k(src, dst),
        13 => dequantize_q5_k(src, dst),
        14 => dequantize_q6_k(src, dst),
        _ => unreachable!("filtered above"),
    }
    true
}

fn f16_at(b: &[u8], i: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[i], b[i + 1]]))
}

/// The four classic 32-element formats differ only in an optional fp16 min
/// and an optional fifth bit per value. Each block is
///   fp16 d; [fp16 m]; [u32 qh]; u8 qs[16]
/// and value j (or j + 16) is the low (or high) nibble of qs[j], plus bit j
/// (or j + 16) of qh as bit 4, times d, then plus m, or centred on 8 (or 16
/// with the fifth bit) when there is no m.
fn dequantize_legacy<const HAS_MIN: bool, const HAS_HIGH: bool>(src: &[u8], dst: &mut [f32]) {
    let bytes = 2 + if HAS_MIN { 2 } else { 0 } + if HAS_HIGH { 4 } else { 0 } + 16;
    let centre = if HAS_MIN {
        0
    } else if HAS_HIGH {
        16
    } else {
        8
    };
    for (p, y) in src.chunks_exact(bytes).zip(dst.as_chunks_mut::<32>().0) {
        let d = f16_at(p, 0);
        let m = if HAS_MIN { f16_at(p, 2) } else { 0.0 };
        let o = if HAS_MIN { 4 } else { 2 };
        let qh = if HAS_HIGH {
            u32::from_le_bytes([p[o], p[o + 1], p[o + 2], p[o + 3]])
        } else {
            0
        };
        let qs = &p[bytes - 16..];
        for j in 0..16 {
            let hi0 = (((qh >> j) << 4) & 0x10) as i32;
            let hi1 = ((qh >> (j + 12)) & 0x10) as i32;
            y[j] = ((i32::from(qs[j] & 0x0f) | hi0) - centre) as f32 * d + m;
            y[j + 16] = ((i32::from(qs[j] >> 4) | hi1) - centre) as f32 * d + m;
        }
    }
}

/// Sub-block `j`'s 6-bit scale and min from a Q4_K/Q5_K block's 12 packed
/// scale bytes (ggml's packing).
fn scale_min_k4(j: usize, q: &[u8]) -> (f32, f32) {
    let (d, m) = if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    };
    (f32::from(d), f32::from(m))
}

/// Q4_K: 144-byte blocks of 256 = { fp16 d, dmin; u8 scales[12]; u8 qs[128] }.
fn dequantize_q4_k(src: &[u8], dst: &mut [f32]) {
    for (b, y) in src.as_chunks::<144>().0.iter().zip(dst.as_chunks_mut::<256>().0) {
        let (d, dmin, scales) = (f16_at(b, 0), f16_at(b, 2), &b[4..16]);
        // Four runs of 64 outputs, each from 32 qs bytes: sub-block 2j from
        // the low nibbles, 2j + 1 from the high ones.
        for (j, (q, y)) in b[16..]
            .as_chunks::<32>()
            .0
            .iter()
            .zip(y.as_chunks_mut::<64>().0)
            .enumerate()
        {
            let (sc, mn) = scale_min_k4(2 * j, scales);
            let (d1, m1) = (d * sc, dmin * mn);
            let (sc, mn) = scale_min_k4(2 * j + 1, scales);
            let (d2, m2) = (d * sc, dmin * mn);
            for l in 0..32 {
                y[l] = d1 * f32::from(q[l] & 0x0f) - m1;
                y[l + 32] = d2 * f32::from(q[l] >> 4) - m2;
            }
        }
    }
}

/// Q5_K: 176-byte blocks of 256 = { fp16 d, dmin; u8 scales[12]; u8 qh[32];
/// u8 qs[128] }, qh carrying each value's fifth bit.
fn dequantize_q5_k(src: &[u8], dst: &mut [f32]) {
    for (b, y) in src.as_chunks::<176>().0.iter().zip(dst.as_chunks_mut::<256>().0) {
        let (d, dmin, scales, qh) = (f16_at(b, 0), f16_at(b, 2), &b[4..16], &b[16..48]);
        let (mut u1, mut u2) = (1u8, 2u8);
        for (j, (q, y)) in b[48..]
            .as_chunks::<32>()
            .0
            .iter()
            .zip(y.as_chunks_mut::<64>().0)
            .enumerate()
        {
            let (sc, mn) = scale_min_k4(2 * j, scales);
            let (d1, m1) = (d * sc, dmin * mn);
            let (sc, mn) = scale_min_k4(2 * j + 1, scales);
            let (d2, m2) = (d * sc, dmin * mn);
            for l in 0..32 {
                let h1 = if qh[l] & u1 != 0 { 16 } else { 0 };
                let h2 = if qh[l] & u2 != 0 { 16 } else { 0 };
                y[l] = d1 * f32::from((q[l] & 0x0f) + h1) - m1;
                y[l + 32] = d2 * f32::from((q[l] >> 4) + h2) - m2;
            }
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

/// Q6_K: 210-byte blocks of 256 = { u8 ql[128]; u8 qh[64]; i8 scales[16];
/// fp16 d }: 4 low bits in ql, 2 high bits in qh, a scale per 16 values.
fn dequantize_q6_k(src: &[u8], dst: &mut [f32]) {
    for (b, y) in src.as_chunks::<210>().0.iter().zip(dst.as_chunks_mut::<256>().0) {
        let d = f16_at(b, 208);
        // Two runs of 128 outputs.
        for (h, y) in y.as_chunks_mut::<128>().0.iter_mut().enumerate() {
            let (ql, qh, sc) = (&b[64 * h..], &b[128 + 32 * h..], &b[192 + 8 * h..]);
            let s = |i: usize| f32::from(sc[i] as i8);
            for l in 0..32 {
                let is = l / 16;
                let q1 = i32::from((ql[l] & 0x0f) | ((qh[l] & 3) << 4)) - 32;
                let q2 = i32::from((ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 3) << 4)) - 32;
                let q3 = i32::from((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                let q4 = i32::from((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l] = d * s(is) * q1 as f32;
                y[l + 32] = d * s(is + 2) * q2 as f32;
                y[l + 64] = d * s(is + 4) * q3 as f32;
                y[l + 96] = d * s(is + 6) * q4 as f32;
            }
        }
    }
}
