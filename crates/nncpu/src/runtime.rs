//! Runtime basics: dtypes, tensor views, aligned buffers and BF16 helpers.

use crate::error::{Result, err};
use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::ptr::NonNull;

pub(crate) const MAX_DIMS: usize = 4;

/// BF16 -> F32 is exact: the BF16 bits are the high half of the F32.
#[inline(always)]
pub fn bf16_to_f32(v: u16) -> f32 {
    f32::from_bits((v as u32) << 16)
}

/// F32 -> BF16, rounding to nearest even.
#[inline(always)]
pub fn f32_to_bf16(f: f32) -> u16 {
    let u = f.to_bits();
    let rounding_bias = 0x7fff + ((u >> 16) & 1);
    (u.wrapping_add(rounding_bias) >> 16) as u16
}

/// IEEE half -> F32, subnormals included.
#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            // Subnormal: renormalise.
            let mut e = 127 - 15 + 1;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | ((e as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

/// Element types of nncpu's in-memory tensors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum DType {
    F32,
    BF16,
    /// Q8_0, split layout: row-major `i8 qs[rows*cols]`, then one BF16 scale
    /// per 32-element block, `u16 scales[rows*cols/32]`, in one allocation.
    Q8_0,
    /// 4-bit, split layout: `u8 qs[rows*cols/2]` packed nibbles, then BF16
    /// `scales[rows*cols/32]`, then BF16 `biases[rows*cols/32]`. A weight is
    /// `scale * q + bias` with `q` in [0, 15]; byte `i` of a block holds
    /// element `i` in its low nibble and element `i + 16` in its high one.
    Q4S,
}

/// A weight view. `data` points into the GGUF mapping or into a buffer the
/// model owns; the view is only valid while that model is alive, which is
/// why tensors never leave the crate.
#[derive(Clone, Debug)]
pub(crate) struct Tensor {
    pub dtype: DType,
    pub n_dims: usize,
    /// Shape, `ne[0]` fastest-varying (the columns of a matrix).
    pub ne: [usize; MAX_DIMS],
    pub data: *const u8,
}

// SAFETY: a tensor is a read-only view of memory its model owns and never
// mutates after load; the model outlives every use of its tensors.
unsafe impl Send for Tensor {}
unsafe impl Sync for Tensor {}

impl Tensor {
    pub fn new(dtype: DType, ne: &[usize], data: *const u8) -> Tensor {
        let mut t = Tensor {
            dtype,
            n_dims: ne.len(),
            ne: [1; MAX_DIMS],
            data,
        };
        t.ne[..ne.len()].copy_from_slice(ne);
        t
    }

    pub fn nelements(&self) -> usize {
        self.ne.iter().product()
    }

    /// Columns: the length of one row, the input length of a gemv.
    #[inline]
    pub fn cols(&self) -> usize {
        self.ne[0]
    }

    /// Rows: the output length of a gemv over this matrix.
    #[inline]
    pub fn rows(&self) -> usize {
        self.ne[1]
    }

    /// Elements of one matrix (`ne[0] * ne[1]`): for an expert stack, of
    /// one expert.
    #[inline]
    fn matrix_elems(&self) -> usize {
        self.ne[0] * self.ne[1]
    }

    #[inline]
    pub fn f32s(&self) -> &[f32] {
        debug_assert_eq!(self.dtype, DType::F32);
        debug_assert_eq!(self.data as usize % 4, 0);
        // SAFETY: an F32 tensor holds nelements() aligned floats.
        unsafe { std::slice::from_raw_parts(self.data as *const f32, self.nelements()) }
    }

    /// A BF16 matrix's weights.
    #[inline]
    pub fn bf16(&self) -> &[u16] {
        debug_assert_eq!(self.dtype, DType::BF16);
        debug_assert_eq!(self.data as usize % 2, 0);
        // SAFETY: a BF16 matrix holds ne[0]*ne[1] aligned u16s.
        unsafe { std::slice::from_raw_parts(self.data as *const u16, self.matrix_elems()) }
    }

    /// A Q8_0 matrix's quants and block scales.
    #[inline]
    pub fn q8(&self) -> (&[i8], &[u16]) {
        debug_assert_eq!(self.dtype, DType::Q8_0);
        let n = self.matrix_elems();
        // SAFETY: the split layout is n quants, then n/32 BF16 scales; n is a
        // multiple of 32 and the base 2-aligned, so the scales are aligned.
        unsafe {
            (
                std::slice::from_raw_parts(self.data as *const i8, n),
                std::slice::from_raw_parts(self.data.add(n) as *const u16, n / 32),
            )
        }
    }

    /// A Q4 matrix's packed nibbles, block scales and block biases.
    #[inline]
    pub fn q4(&self) -> (&[u8], &[u16], &[u16]) {
        debug_assert_eq!(self.dtype, DType::Q4S);
        let n = self.matrix_elems();
        // SAFETY: the split layout is n/2 bytes, then n/32 scales, then n/32
        // biases; n/2 is a multiple of 16, so both tables are aligned.
        unsafe {
            let scales = self.data.add(n / 2) as *const u16;
            (
                std::slice::from_raw_parts(self.data, n / 2),
                std::slice::from_raw_parts(scales, n / 32),
                std::slice::from_raw_parts(scales.add(n / 32), n / 32),
            )
        }
    }
}

/// A zero-initialised, 64-byte-aligned heap buffer whose address never
/// changes, so tensor views can point into it.
pub(crate) struct AlignedBuf {
    ptr: NonNull<u8>,
    len: usize,
}

// SAFETY: AlignedBuf owns its allocation exclusively, like a Vec<u8>.
unsafe impl Send for AlignedBuf {}
unsafe impl Sync for AlignedBuf {}

impl AlignedBuf {
    const ALIGN: usize = 64;

    pub fn zeroed(len: usize) -> Result<AlignedBuf> {
        if len == 0 {
            // A dangling but aligned pointer, as Vec uses for empty buffers.
            let ptr = NonNull::new(Self::ALIGN as *mut u8).unwrap();
            return Ok(AlignedBuf { ptr, len: 0 });
        }
        let layout =
            Layout::from_size_align(len, Self::ALIGN).map_err(|_| err!(OutOfMemory, "cannot allocate {len} bytes"))?;
        // SAFETY: layout has a non-zero size.
        let p = unsafe { alloc_zeroed(layout) };
        let ptr = NonNull::new(p).ok_or_else(|| err!(OutOfMemory, "cannot allocate {} MB", len.div_ceil(1 << 20)))?;
        Ok(AlignedBuf { ptr, len })
    }

    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: we own len initialised bytes.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        if self.len != 0 {
            // SAFETY: allocated in zeroed() with this layout.
            unsafe {
                dealloc(
                    self.ptr.as_ptr(),
                    Layout::from_size_align_unchecked(self.len, Self::ALIGN),
                )
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_round_trip() {
        for &f in &[0.0f32, 1.0, -2.5, 3.140625, 2f32.powi(-100), -(2f32.powi(100))] {
            assert_eq!(bf16_to_f32(f32_to_bf16(f)), f);
        }
        // Round to nearest even.
        assert_eq!(f32_to_bf16(f32::from_bits(0x3f80_8000)), 0x3f80);
        assert_eq!(f32_to_bf16(f32::from_bits(0x3f81_8000)), 0x3f82);
    }

    #[test]
    fn f16_decode() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0001), 5.960_464_5e-8);
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
    }
}
