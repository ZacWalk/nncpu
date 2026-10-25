//! A minimal GGUF v2/v3 reader.
//!
//! The file is memory-mapped and parsed in place: the header, every metadata
//! key/value pair and the tensor descriptor table. Tensor data is never
//! copied; [`GgufFile::tensor_data`] returns slices of the mapping. Every
//! length and count read from the file is checked against the bytes that
//! remain, and the parsed metadata against a memory budget, so a corrupt or
//! hostile file is refused with a message rather than allowed to read past
//! the mapping or allocate far more than its own size.
//!
//! Spec: <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>

use crate::error::{Error, Result, bail, err};
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian

// Per-allocation bounds on what the file may ask for. Anything larger almost
// certainly means a corrupt or hostile input; real models fit easily.
const MAX_STRING_LEN: u64 = 1 << 24;
const MAX_ARRAY_LEN: u64 = 1 << 28;
const MAX_TENSOR_COUNT: u64 = 1 << 20;
const MAX_KV_COUNT: u64 = 1 << 20;
// Real files nest at most one level; each level costs 12 bytes of file but a
// recursion frame, so without a bound nested headers overflow the stack.
const MAX_ARRAY_DEPTH: u32 = 8;
// Metadata values together may take about as much memory as the file is
// long, or 256 MiB for a small file.
const MIN_METADATA_BUDGET: u64 = 256 << 20;

/// GGUF metadata value types.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GgufType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    F32,
    Bool,
    String,
    Array,
    U64,
    I64,
    F64,
}

impl GgufType {
    fn from_u32(v: u32) -> Option<GgufType> {
        Some(match v {
            0 => GgufType::U8,
            1 => GgufType::I8,
            2 => GgufType::U16,
            3 => GgufType::I16,
            4 => GgufType::U32,
            5 => GgufType::I32,
            6 => GgufType::F32,
            7 => GgufType::Bool,
            8 => GgufType::String,
            9 => GgufType::Array,
            10 => GgufType::U64,
            11 => GgufType::I64,
            12 => GgufType::F64,
            _ => return None,
        })
    }

    /// The fewest bytes one value of this type occupies in the file.
    fn min_encoded_size(self) -> u64 {
        match self {
            GgufType::U16 | GgufType::I16 => 2,
            GgufType::U32 | GgufType::I32 | GgufType::F32 => 4,
            GgufType::U64 | GgufType::I64 | GgufType::F64 | GgufType::String => 8,
            GgufType::Array => 12,
            _ => 1,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            GgufType::U8 => "u8",
            GgufType::I8 => "i8",
            GgufType::U16 => "u16",
            GgufType::I16 => "i16",
            GgufType::U32 => "u32",
            GgufType::I32 => "i32",
            GgufType::F32 => "f32",
            GgufType::Bool => "bool",
            GgufType::String => "str",
            GgufType::Array => "arr",
            GgufType::U64 => "u64",
            GgufType::I64 => "i64",
            GgufType::F64 => "f64",
        }
    }
}

/// A metadata value.
#[derive(Clone, Debug, PartialEq)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    /// The element type and the elements.
    Array(GgufType, Vec<GgufValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl GgufValue {
    pub fn ty(&self) -> GgufType {
        match self {
            GgufValue::U8(_) => GgufType::U8,
            GgufValue::I8(_) => GgufType::I8,
            GgufValue::U16(_) => GgufType::U16,
            GgufValue::I16(_) => GgufType::I16,
            GgufValue::U32(_) => GgufType::U32,
            GgufValue::I32(_) => GgufType::I32,
            GgufValue::F32(_) => GgufType::F32,
            GgufValue::Bool(_) => GgufType::Bool,
            GgufValue::Str(_) => GgufType::String,
            GgufValue::Array(..) => GgufType::Array,
            GgufValue::U64(_) => GgufType::U64,
            GgufValue::I64(_) => GgufType::I64,
            GgufValue::F64(_) => GgufType::F64,
        }
    }

    /// An integer value of any width and signedness (not a bool).
    pub fn as_int(&self) -> Option<i64> {
        Some(match *self {
            GgufValue::U8(v) => v as i64,
            GgufValue::U16(v) => v as i64,
            GgufValue::U32(v) => v as i64,
            GgufValue::U64(v) => i64::try_from(v).ok()?,
            GgufValue::I8(v) => v as i64,
            GgufValue::I16(v) => v as i64,
            GgufValue::I32(v) => v as i64,
            GgufValue::I64(v) => v,
            _ => return None,
        })
    }

    /// A float value (F32 or F64).
    pub fn as_float(&self) -> Option<f64> {
        match *self {
            GgufValue::F32(v) => Some(v as f64),
            GgufValue::F64(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            GgufValue::Bool(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufValue::Str(s) => Some(s),
            _ => None,
        }
    }

    /// An array's element type and elements.
    pub fn as_array(&self) -> Option<(GgufType, &[GgufValue])> {
        match self {
            GgufValue::Array(t, v) => Some((*t, v)),
            _ => None,
        }
    }

    /// Any numeric value as f64, else 0 (the lenient read vocabularies use).
    pub fn to_f64_or_zero(&self) -> f64 {
        self.as_float()
            .or_else(|| self.as_int().map(|v| v as f64))
            .unwrap_or(0.0)
    }

    /// Any integer or bool as i64, else 0.
    pub fn to_i64_or_zero(&self) -> i64 {
        match *self {
            GgufValue::Bool(b) => b as i64,
            _ => self.as_int().unwrap_or(0),
        }
    }
}

/// One metadata key/value pair.
#[derive(Clone, Debug)]
pub struct GgufKv {
    pub key: String,
    pub value: GgufValue,
}

/// One tensor descriptor.
#[derive(Clone, Debug)]
pub struct GgufTensorInfo {
    pub name: String,
    pub n_dims: u32,
    /// Shape, `ne[0]` fastest-varying; unused dimensions are 1.
    pub ne: [u64; 4],
    /// The ggml type code (0 = F32, 1 = F16, 8 = Q8_0, 12 = Q4_K, 30 = BF16...).
    pub ggml_type: u32,
    /// Offset within the tensor data block.
    pub offset: u64,
}

impl GgufTensorInfo {
    /// Total element count (the product of all dimensions).
    pub fn nelements(&self) -> u64 {
        self.ne[..self.n_dims as usize].iter().product()
    }

    /// Storage size in bytes, aware of block quantization; 0 for types this
    /// build does not know.
    pub fn nbytes(&self) -> u64 {
        match ggml_block(self.ggml_type) {
            Some((bytes, elems)) => self.nelements() / elems as u64 * bytes as u64,
            None => 0,
        }
    }
}

/// A parsed GGUF file, optionally still mapped.
pub struct GgufFile {
    pub version: u32,
    pub alignment: u64,
    /// Absolute file offset of the tensor data block.
    pub data_offset: u64,
    pub kv: Vec<GgufKv>,
    pub tensors: Vec<GgufTensorInfo>,
    /// Size of the file in bytes.
    pub file_size: u64,
    map: Option<Mmap>,
}

impl std::fmt::Debug for GgufFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GgufFile")
            .field("version", &self.version)
            .field("kv", &self.kv.len())
            .field("tensors", &self.tensors.len())
            .field("mapped", &self.map.is_some())
            .finish()
    }
}

impl GgufFile {
    /// Map `path` read-only and parse its header. The mapping stays valid as
    /// long as the `GgufFile`; [`tensor_data`](Self::tensor_data) points into it.
    pub fn open(path: impl AsRef<Path>) -> Result<GgufFile> {
        let path = path.as_ref();
        let map = map_file(path)?;
        let mut f = parse(&map, path)?;
        f.map = Some(map);
        Ok(f)
    }

    /// Parse the header, metadata and tensor descriptors only; the file is
    /// not left mapped, so `tensor_data` returns `None` afterwards.
    pub fn load_header(path: impl AsRef<Path>) -> Result<GgufFile> {
        let mut f = GgufFile::open(path)?;
        f.map = None;
        Ok(f)
    }

    /// Parse GGUF bytes already in memory (tests, tools). Tensor data is not
    /// available from the result.
    pub fn parse_bytes(bytes: &[u8], name: &str) -> Result<GgufFile> {
        parse(bytes, Path::new(name))
    }

    pub fn is_mapped(&self) -> bool {
        self.map.is_some()
    }

    /// The whole mapped file, when mapped.
    pub fn mapped(&self) -> Option<&[u8]> {
        self.map.as_deref()
    }

    /// Ask the OS to start reading the whole mapped file into memory.
    pub fn prefetch(&self) {
        if let Some(map) = &self.map {
            prefetch(map);
        }
    }

    /// Find a metadata value by exact key.
    pub fn get(&self, key: &str) -> Option<&GgufValue> {
        self.kv.iter().find(|kv| kv.key == key).map(|kv| &kv.value)
    }

    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(GgufValue::as_str)
    }

    /// Find a tensor by exact name.
    pub fn find_tensor(&self, name: &str) -> Option<usize> {
        self.tensors.iter().position(|t| t.name == name)
    }

    /// Tensor `i`'s raw bytes within the mapped data block, or `None` when
    /// the file is not mapped or the bytes do not lie inside it.
    pub fn tensor_data(&self, i: usize) -> Option<&[u8]> {
        let map = self.map.as_ref()?;
        let t = self.tensors.get(i)?;
        let off = self.data_offset.checked_add(t.offset)?;
        let size = map.len() as u64;
        if off >= size {
            return None;
        }
        let nbytes = t.nbytes();
        if nbytes > size - off {
            return None;
        }
        Some(&map[off as usize..(off + nbytes) as usize])
    }
}

/// {bytes per block, elements per block} for the ggml types we know.
pub(crate) fn ggml_block(t: u32) -> Option<(u32, u32)> {
    Some(match t {
        0 => (4, 1),      // F32
        1 => (2, 1),      // F16
        2 => (18, 32),    // Q4_0
        3 => (20, 32),    // Q4_1
        6 => (22, 32),    // Q5_0
        7 => (24, 32),    // Q5_1
        8 => (34, 32),    // Q8_0
        9 => (40, 32),    // Q8_1
        10 => (84, 256),  // Q2_K
        11 => (110, 256), // Q3_K
        12 => (144, 256), // Q4_K
        13 => (176, 256), // Q5_K
        14 => (210, 256), // Q6_K
        15 => (292, 256), // Q8_K
        24 => (1, 1),     // I8
        25 => (2, 1),     // I16
        26 => (4, 1),     // I32
        27 => (8, 1),     // I64
        28 => (8, 1),     // F64
        30 => (2, 1),     // BF16
        _ => return None,
    })
}

/// Human-readable name for a ggml tensor type code ("?(n)" when unknown).
pub fn ggml_type_name(t: u32) -> String {
    let name = match t {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        15 => "Q8_K",
        16 => "IQ2_XXS",
        17 => "IQ2_XS",
        18 => "IQ3_XXS",
        19 => "IQ1_S",
        20 => "IQ4_NL",
        21 => "IQ3_S",
        22 => "IQ2_S",
        23 => "IQ4_XS",
        24 => "I8",
        25 => "I16",
        26 => "I32",
        27 => "I64",
        28 => "F64",
        29 => "IQ1_M",
        30 => "BF16",
        31 => "Q4_0_4_4",
        32 => "Q4_0_4_8",
        33 => "Q4_0_8_8",
        34 => "TQ1_0",
        35 => "TQ2_0",
        _ => return format!("?({t})"),
    };
    name.to_string()
}

fn map_file(path: &Path) -> Result<Mmap> {
    let mut opts = std::fs::OpenOptions::new();
    opts.read(true);
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_FLAG_RANDOM_ACCESS: u32 = 0x1000_0000;
        opts.custom_flags(FILE_FLAG_RANDOM_ACCESS);
    }
    let file: File = opts.open(path).map_err(|e| io_error("open", path, &e))?;
    let len = file.metadata().map_err(|e| io_error("size", path, &e))?.len();
    if len == 0 {
        bail!(BadFormat, "'{}' is empty", path.display());
    }
    // SAFETY: the mapping is read-only. Like every mmap-based loader we rely
    // on the file not being truncated while it is mapped.
    unsafe { Mmap::map(&file) }.map_err(|e| io_error("map", path, &e))
}

fn io_error(what: &str, path: &Path, e: &std::io::Error) -> Error {
    err!(Io, "cannot {what} '{}': {e}", path.display())
}

#[cfg(windows)]
fn prefetch(map: &Mmap) {
    use std::ffi::c_void;
    #[repr(C)]
    struct MemoryRangeEntry {
        virtual_address: *mut c_void,
        number_of_bytes: usize,
    }
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut c_void;
        fn PrefetchVirtualMemory(process: *mut c_void, n: usize, entries: *const MemoryRangeEntry, flags: u32) -> i32;
    }
    let entry = MemoryRangeEntry {
        virtual_address: map.as_ptr() as *mut c_void,
        number_of_bytes: map.len(),
    };
    // SAFETY: the range is our own live mapping; the call is only a hint.
    unsafe {
        PrefetchVirtualMemory(GetCurrentProcess(), 1, &entry, 0);
    }
}

#[cfg(unix)]
fn prefetch(map: &Mmap) {
    let _ = map.advise(memmap2::Advice::WillNeed);
}

#[cfg(not(any(windows, unix)))]
fn prefetch(_map: &Mmap) {}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
    path: &'a Path,
    depth: u32,
    values_left: u64,
}

impl<'a> Reader<'a> {
    fn remaining(&self) -> u64 {
        (self.data.len() - self.pos) as u64
    }

    fn fail(&self, what: &str) -> Error {
        err!(BadFormat, "'{}' is not a valid GGUF file: {what}", self.path.display())
    }

    fn bytes(&mut self, n: u64) -> Result<&'a [u8]> {
        if n > self.remaining() {
            return Err(self.fail("it ends early (truncated?)"));
        }
        let s = &self.data[self.pos..self.pos + n as usize];
        self.pos += n as usize;
        Ok(s)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
        Ok(self.bytes(N as u64)?.try_into().unwrap())
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.array::<1>()?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.array()?))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.array()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    fn string(&mut self) -> Result<String> {
        let len = self.u64()?;
        if len > MAX_STRING_LEN {
            return Err(self.fail("a string is implausibly long"));
        }
        let b = self.bytes(len)?;
        // Vocabularies hold the odd invalid sequence; keep what we can.
        Ok(String::from_utf8_lossy(b).into_owned())
    }

    fn value(&mut self, ty: GgufType) -> Result<GgufValue> {
        if self.values_left == 0 {
            return Err(self.fail("its metadata would take implausibly much memory"));
        }
        self.values_left -= 1;
        Ok(match ty {
            GgufType::U8 => GgufValue::U8(self.u8()?),
            GgufType::I8 => GgufValue::I8(self.u8()? as i8),
            GgufType::U16 => GgufValue::U16(self.u16()?),
            GgufType::I16 => GgufValue::I16(self.u16()? as i16),
            GgufType::U32 => GgufValue::U32(self.u32()?),
            GgufType::I32 => GgufValue::I32(self.u32()? as i32),
            GgufType::U64 => GgufValue::U64(self.u64()?),
            GgufType::I64 => GgufValue::I64(self.u64()? as i64),
            GgufType::F32 => GgufValue::F32(f32::from_bits(self.u32()?)),
            GgufType::F64 => GgufValue::F64(f64::from_bits(self.u64()?)),
            GgufType::Bool => GgufValue::Bool(self.u8()? != 0),
            GgufType::String => GgufValue::Str(self.string()?),
            GgufType::Array => {
                if self.depth >= MAX_ARRAY_DEPTH {
                    return Err(self.fail("arrays are nested too deeply"));
                }
                self.depth += 1;
                let v = self.array_value();
                self.depth -= 1;
                v?
            }
        })
    }

    fn array_value(&mut self) -> Result<GgufValue> {
        let elem = self.u32()?;
        let len = self.u64()?;
        let elem = GgufType::from_u32(elem).ok_or_else(|| self.fail("a metadata value has an unknown type"))?;
        // Every element takes at least this many bytes of the file, so a
        // length the rest of the file cannot hold is corrupt or hostile.
        if len > MAX_ARRAY_LEN || len > self.remaining() / elem.min_encoded_size() {
            return Err(self.fail("an array is longer than the file can hold"));
        }
        if len > self.values_left {
            return Err(self.fail("its metadata would take implausibly much memory"));
        }
        let mut out = Vec::with_capacity(len.min(65536) as usize);
        for _ in 0..len {
            out.push(self.value(elem)?);
        }
        Ok(GgufValue::Array(elem, out))
    }
}

fn parse(data: &[u8], path: &Path) -> Result<GgufFile> {
    let mut r = Reader {
        data,
        pos: 0,
        path,
        depth: 0,
        values_left: (data.len() as u64).max(MIN_METADATA_BUDGET) / std::mem::size_of::<GgufValue>() as u64,
    };

    if data.len() < 4 || r.u32()? != GGUF_MAGIC {
        bail!(BadFormat, "'{}' is not a GGUF file", path.display());
    }
    let version = r.u32()?;
    if version != 2 && version != 3 {
        bail!(
            UnsupportedModel,
            "'{}' is GGUF version {version}; nncpu reads versions 2 and 3",
            path.display()
        );
    }

    let tensor_count = r.u64()?;
    let kv_count = r.u64()?;
    // Besides the hard caps, each KV needs at least 13 bytes and each tensor
    // descriptor 32, so counts the rest of the file cannot hold are refused
    // before anything is sized.
    if tensor_count > MAX_TENSOR_COUNT || kv_count > MAX_KV_COUNT || kv_count * 13 + tensor_count * 32 > r.remaining() {
        return Err(r.fail("its tensor and metadata counts do not fit in the file"));
    }

    let mut kv = Vec::with_capacity(kv_count as usize);
    for _ in 0..kv_count {
        let key = r.string()?;
        let ty = r.u32()?;
        let ty = GgufType::from_u32(ty).ok_or_else(|| r.fail("a metadata value has an unknown type"))?;
        let value = r.value(ty)?;
        kv.push(GgufKv { key, value });
    }

    let mut alignment = 32;
    if let Some(GgufValue::U32(a)) = kv.iter().find(|kv| kv.key == "general.alignment").map(|kv| &kv.value) {
        alignment = if *a != 0 { *a as u64 } else { 32 };
    }

    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = r.string()?;
        let n_dims = r.u32()?;
        if n_dims == 0 || n_dims > 4 {
            return Err(r.fail("a tensor has an invalid number of dimensions"));
        }
        let mut ne = [1u64; 4];
        for d in ne.iter_mut().take(n_dims as usize) {
            *d = r.u64()?;
        }
        // Dimensions are kept as i32-sized values downstream, and byte counts
        // derive from the element count, so a dimension beyond i32::MAX or a
        // count whose byte size (up to 8 bytes an element) would wrap lets a
        // small file describe a huge tensor.
        let max_elements = i64::MAX as u64 / 8;
        let mut elements = 1u64;
        for &d in &ne[..n_dims as usize] {
            if d > i32::MAX as u64 || (d != 0 && elements > max_elements / d) {
                return Err(r.fail("a tensor has an impossible shape"));
            }
            elements *= d;
        }
        let ggml_type = r.u32()?;
        let offset = r.u64()?;
        tensors.push(GgufTensorInfo {
            name,
            n_dims,
            ne,
            ggml_type,
            offset,
        });
    }

    // The tensor data block starts at the next multiple of `alignment`.
    let pos = r.pos as u64;
    let data_offset = pos.div_ceil(alignment) * alignment;
    Ok(GgufFile {
        version,
        alignment,
        data_offset,
        kv,
        tensors,
        file_size: data.len() as u64,
        map: None,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::error::ErrorKind;

    /// A tiny GGUF writer for tests: metadata and F32 tensors.
    #[derive(Default)]
    pub(crate) struct GgufWriter {
        kv: Vec<(String, Vec<u8>)>,
        tensors: Vec<(String, Vec<u64>, u32, Vec<u8>)>,
    }

    fn put_str(out: &mut Vec<u8>, s: &str) {
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }

    impl GgufWriter {
        pub fn u32(&mut self, key: &str, v: u32) -> &mut Self {
            let mut b = 4u32.to_le_bytes().to_vec();
            b.extend_from_slice(&v.to_le_bytes());
            self.kv.push((key.into(), b));
            self
        }
        pub fn f32(&mut self, key: &str, v: f32) -> &mut Self {
            let mut b = 6u32.to_le_bytes().to_vec();
            b.extend_from_slice(&v.to_le_bytes());
            self.kv.push((key.into(), b));
            self
        }
        pub fn bool(&mut self, key: &str, v: bool) -> &mut Self {
            let mut b = 7u32.to_le_bytes().to_vec();
            b.push(v as u8);
            self.kv.push((key.into(), b));
            self
        }
        pub fn str(&mut self, key: &str, v: &str) -> &mut Self {
            let mut b = 8u32.to_le_bytes().to_vec();
            put_str(&mut b, v);
            self.kv.push((key.into(), b));
            self
        }
        pub fn strs(&mut self, key: &str, v: &[&str]) -> &mut Self {
            let mut b = 9u32.to_le_bytes().to_vec();
            b.extend_from_slice(&8u32.to_le_bytes());
            b.extend_from_slice(&(v.len() as u64).to_le_bytes());
            for s in v {
                put_str(&mut b, s);
            }
            self.kv.push((key.into(), b));
            self
        }
        pub fn f32s(&mut self, key: &str, v: &[f32]) -> &mut Self {
            let mut b = 9u32.to_le_bytes().to_vec();
            b.extend_from_slice(&6u32.to_le_bytes());
            b.extend_from_slice(&(v.len() as u64).to_le_bytes());
            for x in v {
                b.extend_from_slice(&x.to_le_bytes());
            }
            self.kv.push((key.into(), b));
            self
        }
        pub fn i32s(&mut self, key: &str, v: &[i32]) -> &mut Self {
            let mut b = 9u32.to_le_bytes().to_vec();
            b.extend_from_slice(&5u32.to_le_bytes());
            b.extend_from_slice(&(v.len() as u64).to_le_bytes());
            for x in v {
                b.extend_from_slice(&x.to_le_bytes());
            }
            self.kv.push((key.into(), b));
            self
        }
        pub fn bools(&mut self, key: &str, v: &[bool]) -> &mut Self {
            let mut b = 9u32.to_le_bytes().to_vec();
            b.extend_from_slice(&7u32.to_le_bytes());
            b.extend_from_slice(&(v.len() as u64).to_le_bytes());
            b.extend(v.iter().map(|&x| x as u8));
            self.kv.push((key.into(), b));
            self
        }
        /// A tensor of raw bytes with ggml type `ty`.
        pub fn tensor_raw(&mut self, name: &str, ne: &[u64], ty: u32, data: Vec<u8>) -> &mut Self {
            self.tensors.push((name.into(), ne.to_vec(), ty, data));
            self
        }
        pub fn tensor_f32(&mut self, name: &str, ne: &[u64], data: &[f32]) -> &mut Self {
            let bytes = data.iter().flat_map(|x| x.to_le_bytes()).collect();
            self.tensor_raw(name, ne, 0, bytes)
        }

        pub fn bytes(&self) -> Vec<u8> {
            let mut out = Vec::new();
            out.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
            out.extend_from_slice(&3u32.to_le_bytes());
            out.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
            out.extend_from_slice(&(self.kv.len() as u64).to_le_bytes());
            for (k, v) in &self.kv {
                put_str(&mut out, k);
                out.extend_from_slice(v);
            }
            let mut offset = 0u64;
            let mut offsets = Vec::new();
            for (name, ne, ty, data) in &self.tensors {
                put_str(&mut out, name);
                out.extend_from_slice(&(ne.len() as u32).to_le_bytes());
                for d in ne {
                    out.extend_from_slice(&d.to_le_bytes());
                }
                out.extend_from_slice(&ty.to_le_bytes());
                out.extend_from_slice(&offset.to_le_bytes());
                offsets.push(offset);
                offset += (data.len() as u64).div_ceil(32) * 32;
            }
            while out.len() % 32 != 0 {
                out.push(0);
            }
            let base = out.len();
            for ((_, _, _, data), off) in self.tensors.iter().zip(offsets) {
                out.resize(base + off as usize, 0);
                out.extend_from_slice(data);
            }
            out
        }

        pub fn write(&self, path: &Path) {
            std::fs::write(path, self.bytes()).unwrap();
        }
    }

    pub(crate) fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("nncpu-test-{}-{name}", std::process::id()))
    }

    #[test]
    fn round_trip() {
        let mut w = GgufWriter::default();
        w.str("general.architecture", "llama")
            .u32("llama.block_count", 2)
            .f32("llama.rope.freq_base", 10000.0)
            .strs("tokenizer.ggml.tokens", &["a", "b"])
            .tensor_f32("w", &[4, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let path = temp_path("round_trip.gguf");
        w.write(&path);
        let f = GgufFile::open(&path).unwrap();
        assert_eq!(f.version, 3);
        assert_eq!(f.get_str("general.architecture"), Some("llama"));
        assert_eq!(f.get("llama.block_count").and_then(GgufValue::as_int), Some(2));
        assert_eq!(
            f.get("llama.rope.freq_base").and_then(GgufValue::as_float),
            Some(10000.0)
        );
        let i = f.find_tensor("w").unwrap();
        let data = f.tensor_data(i).unwrap();
        assert_eq!(data.len(), 32);
        assert_eq!(f32::from_le_bytes(data[4..8].try_into().unwrap()), 2.0);
        drop(f);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn rejects_bad_input() {
        assert_eq!(
            GgufFile::parse_bytes(b"GGUX\x03\0\0\0", "x").unwrap_err().kind(),
            ErrorKind::BadFormat
        );
        // A truncated header.
        let w = GgufWriter::default();
        let mut b = w.bytes();
        b.truncate(10);
        assert_eq!(GgufFile::parse_bytes(&b, "x").unwrap_err().kind(), ErrorKind::BadFormat);
        // A missing file.
        assert_eq!(
            GgufFile::open("definitely/not/here.gguf").unwrap_err().kind(),
            ErrorKind::Io
        );
    }

    /// A GGUF holding one KV whose value has type `ty` and the given bytes.
    fn one_kv(key: &str, ty: u32, value: &[u8]) -> Vec<u8> {
        let mut b = GGUF_MAGIC.to_le_bytes().to_vec();
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes()); // tensors
        b.extend_from_slice(&1u64.to_le_bytes()); // KVs
        put_str(&mut b, key);
        b.extend_from_slice(&ty.to_le_bytes());
        b.extend_from_slice(value);
        b
    }

    fn array_header(elem: u32, len: u64) -> Vec<u8> {
        let mut b = elem.to_le_bytes().to_vec();
        b.extend_from_slice(&len.to_le_bytes());
        b
    }

    // Paths are UTF-8 throughout, so a file whose name is outside the Windows
    // ANSI code page must open.
    #[test]
    fn utf8_path() {
        let path = temp_path("\u{e9}\u{20ac}\u{6f22}.gguf"); // é € 漢
        std::fs::write(&path, one_kv("general.alignment", 4, &64u32.to_le_bytes())).unwrap();
        let g = GgufFile::load_header(&path).unwrap();
        assert_eq!((g.version, g.kv.len(), g.alignment, g.tensors.len()), (3, 1, 64, 0));
        std::fs::remove_file(&path).unwrap();
    }

    // A few bytes must not be able to claim gigabytes: an array whose length
    // the rest of the file cannot hold is refused before anything is sized.
    #[test]
    fn rejects_oversized_array() {
        let mut value = array_header(0, 1 << 27);
        value.push(7); // the only element actually present
        let err = GgufFile::parse_bytes(&one_kv("bogus", 9, &value), "x").unwrap_err();
        assert_eq!(err.kind(), ErrorKind::BadFormat);
    }

    // Each level of a nested array is a recursion frame for 12 bytes of file,
    // so a deep nest must be refused rather than overflow the stack, while an
    // array of arrays still loads.
    #[test]
    fn rejects_deep_array_nesting() {
        let mut deep = Vec::new();
        for _ in 1..100000 {
            deep.extend(array_header(9, 1));
        }
        deep.extend(array_header(0, 1));
        deep.push(7);
        assert!(GgufFile::parse_bytes(&one_kv("deep", 9, &deep), "x").is_err());

        let mut two = array_header(9, 1);
        two.extend(array_header(0, 1));
        two.push(7);
        let g = GgufFile::parse_bytes(&one_kv("two", 9, &two), "x").unwrap();
        let (_, outer) = g.kv[0].value.as_array().unwrap();
        let (_, inner) = outer[0].as_array().unwrap();
        assert_eq!(inner, &[GgufValue::U8(7)]);
    }

    // Each metadata value is far larger in memory than its one byte of file,
    // so the parser budgets them: an array past the budget is refused before
    // any of it is read.
    #[test]
    fn rejects_metadata_bomb() {
        let length = (MIN_METADATA_BUDGET / std::mem::size_of::<GgufValue>() as u64) + 1;
        let mut value = array_header(0, length);
        value.resize(value.len() + length as usize, 7);
        let err = GgufFile::parse_bytes(&one_kv("bomb", 9, &value), "x").unwrap_err();
        assert_eq!(err.kind(), ErrorKind::BadFormat);
    }

    /// One F32 tensor of shape `dims` and `data_bytes` of zeroed data.
    fn one_tensor(dims: &[u64], data_bytes: usize) -> Vec<u8> {
        let mut b = GGUF_MAGIC.to_le_bytes().to_vec();
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes()); // tensors
        b.extend_from_slice(&0u64.to_le_bytes()); // KVs
        put_str(&mut b, "t");
        b.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for d in dims {
            b.extend_from_slice(&d.to_le_bytes());
        }
        b.extend_from_slice(&0u32.to_le_bytes()); // F32
        b.extend_from_slice(&0u64.to_le_bytes()); // offset
        b.resize(b.len().div_ceil(32) * 32 + data_bytes, 0);
        b
    }

    // Dimensions are kept as i32-sized values and byte counts derive from the
    // element count, so a dimension past i32::MAX or a product that wraps is
    // refused at load: [2^62 + 1, 4] would otherwise count as 4 elements.
    #[test]
    fn rejects_impossible_shapes() {
        let bad: [&[u64]; 4] = [
            &[1 << 32, 1 << 32],
            &[(1 << 62) + 1, 4],
            &[1 << 30, 1 << 30, 1 << 30, 1 << 30],
            &[1 << 21, 1 << 21, 1 << 20], // 2^62 F32 elements: 2^64 bytes wraps to 0
        ];
        for dims in bad {
            assert!(GgufFile::parse_bytes(&one_tensor(dims, 64), "x").is_err(), "{dims:?}");
        }
        let g = GgufFile::parse_bytes(&one_tensor(&[4, 2], 32), "x").unwrap();
        assert_eq!(g.tensors[0].nelements(), 8);
    }
}
