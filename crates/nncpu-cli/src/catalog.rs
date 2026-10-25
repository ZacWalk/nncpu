//! The models nncpu offers to download.
//!
//! `bytes` is the exact file size: it drives the progress bar and is how a
//! finished download is verified. The entries are in the order the menu
//! shows them; the first is the recommended default. Drafters (mtp-*) are
//! left out of the menu; nncpu suggests them by name for their model.

#[derive(Debug)]
pub struct CatalogEntry {
    /// What -m and --download take, e.g. "gemma-3-1b-q6k".
    pub name: &'static str,
    /// The file it is saved as.
    pub file: &'static str,
    pub url: &'static str,
    pub bytes: u64,
    pub description: &'static str,
}

const fn model(
    name: &'static str,
    file: &'static str,
    url: &'static str,
    bytes: u64,
    description: &'static str,
) -> CatalogEntry {
    CatalogEntry {
        name,
        file,
        url,
        bytes,
        description,
    }
}

pub const CATALOG: &[CatalogEntry] = &[
    model(
        "gemma-3-1b-q6k",
        "gemma-3-1b-it-Q6_K.gguf",
        "https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q6_K.gguf",
        1011738656,
        "Gemma 3 1B instruct, 6-bit: small and quick, the best first model",
    ),
    model(
        "gemma-4-e2b-q4km",
        "gemma-4-E2B-it-Q4_K_M.gguf",
        "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf",
        3106738272,
        "Gemma 4 E2B instruct, 4-bit: Google's newest small model, smarter, ~2x slower",
    ),
    model(
        "gemma-3-4b-q4km",
        "gemma-3-4b-it-Q4_K_M.gguf",
        "https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_M.gguf",
        2489894016,
        "Gemma 3 4B instruct, 4-bit: noticeably smarter, about 4x slower",
    ),
    model(
        "gemma-4-e4b-q4km",
        "gemma-4-E4B-it-Q4_K_M.gguf",
        "https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf",
        4977171584,
        "Gemma 4 E4B instruct, 4-bit: stronger again, about 5x slower",
    ),
    model(
        "gemma-4-26b-a4b-q4km",
        "gemma-4-26B-A4B-it-Q4_K_M.gguf",
        "https://huggingface.co/lmstudio-community/gemma-4-26B-A4B-it-GGUF/resolve/main/gemma-4-26B-A4B-it-Q4_K_M.gguf",
        16796017248,
        "Gemma 4 26B A4B (mixture of experts), 4-bit: the strongest; run with --q4 and 16 GB of RAM",
    ),
    model(
        "llama-3.2-1b-q6k",
        "Llama-3.2-1B-Instruct-Q6_K.gguf",
        "https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q6_K.gguf",
        1021800480,
        "Llama 3.2 1B instruct, 6-bit: Meta's small model",
    ),
    model(
        "qwen3.5-0.8b-q8",
        "Qwen3.5-0.8B-Q8_0.gguf",
        "https://huggingface.co/ggml-org/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q8_0.gguf",
        833592096,
        "Qwen 3.5 0.8B, 8-bit: Alibaba's small hybrid (Gated DeltaNet) model, quick",
    ),
    model(
        "qwen3.5-4b-q4km",
        "Qwen3.5-4B-Q4_K_M.gguf",
        "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf",
        2740937888,
        "Qwen 3.5 4B, 4-bit: a much stronger Qwen, about 5x slower",
    ),
    model(
        "qwen3-4b-2507-q4km",
        "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        2497281120,
        "Qwen3 4B Instruct 2507, 4-bit: a strong all-attention Qwen that answers without thinking",
    ),
    model(
        "qwen3.8-27b-q4km",
        "Qwen3.8-27B-Q4_K_M.gguf",
        "https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-Q4_K_M.gguf",
        18973870528,
        "Qwen 3.8 27B, 4-bit: Alibaba's newest open dense model; run with --q4 and 24 GB of RAM",
    ),
    model(
        "gemma-3-270m-q8",
        "gemma-3-270m-it-Q8_0.gguf",
        "https://huggingface.co/ggml-org/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-Q8_0.gguf",
        291545600,
        "Gemma 3 270M instruct, 8-bit: tiny and very fast, for simple tasks",
    ),
    model(
        "functiongemma-270m-q8",
        "functiongemma-270m-it-q8_0.gguf",
        "https://huggingface.co/ggml-org/functiongemma-270m-it-GGUF/resolve/main/functiongemma-270m-it-q8_0.gguf",
        291557792,
        "FunctionGemma 270M: turns requests into function calls (examples/function_call.rs)",
    ),
    model(
        "translategemma-4b-q4km",
        "translategemma-4b-it.Q4_K_M.gguf",
        "https://huggingface.co/mradermacher/translategemma-4b-it-GGUF/resolve/main/translategemma-4b-it.Q4_K_M.gguf",
        2489909760,
        "TranslateGemma 4B, 4-bit: translates between 55 languages (--translate en:de)",
    ),
    model(
        "embeddinggemma-300m-q8",
        "embeddinggemma-300M-Q8_0.gguf",
        "https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF/resolve/main/embeddinggemma-300M-Q8_0.gguf",
        333590944,
        "EmbeddingGemma 300M: text to vectors, for search and similarity",
    ),
    model(
        "embeddinggemma-2-q8",
        "embeddinggemma-2-Q8_0.gguf",
        "https://huggingface.co/ggml-org/embeddinggemma-2-GGUF/resolve/main/embeddinggemma-2-Q8_0.gguf",
        309855456,
        "EmbeddingGemma 2: newer text embeddings, for texts up to 8K tokens",
    ),
    model(
        "gemma-3-1b-bf16",
        "gemma-3-1b-it-BF16.gguf",
        "https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-BF16.gguf",
        2006573344,
        "Gemma 3 1B instruct, 16-bit source weights (for --bf16)",
    ),
    model(
        "mtp-gemma-4-e2b-q8",
        "mtp-gemma-4-E2B-it-Q8_0.gguf",
        "https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF/resolve/main/mtp-gemma-4-E2B-it-Q8_0.gguf",
        97817696,
        "MTP drafter for Gemma 4 E2B: faster replies, used automatically",
    ),
    model(
        "mtp-gemma-4-e4b-q8",
        "mtp-gemma-4-E4B-it-Q8_0.gguf",
        "https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/resolve/main/mtp-gemma-4-E4B-it-Q8_0.gguf",
        98653280,
        "MTP drafter for Gemma 4 E4B: faster replies, used automatically",
    ),
    model(
        "mtp-gemma-4-26b-a4b-q8",
        "mtp-gemma-4-26B-A4B-it-Q8_0.gguf",
        "https://huggingface.co/ggml-org/gemma-4-26B-A4B-it-GGUF/resolve/main/mtp-gemma-4-26B-A4B-it-Q8_0.gguf",
        461766784,
        "MTP drafter for Gemma 4 26B A4B: faster replies, used automatically",
    ),
];

/// The entry whose name or file name is `name`.
pub fn find_catalog_entry(name: &str) -> Option<&'static CatalogEntry> {
    CATALOG.iter().find(|e| e.name == name || e.file == name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn lookup_by_name_or_file() {
        let e = find_catalog_entry("gemma-3-1b-q6k").unwrap();
        assert_eq!(e.file, "gemma-3-1b-it-Q6_K.gguf");
        assert_eq!(e.bytes, 1011738656);
        let by_file = find_catalog_entry("mtp-gemma-4-E2B-it-Q8_0.gguf").unwrap();
        assert_eq!(by_file.name, "mtp-gemma-4-e2b-q8");
        assert_eq!(by_file.bytes, 97817696);
        assert!(
            find_catalog_entry("Gemma-3-1b-q6k").is_none(),
            "names are matched exactly"
        );
        assert!(find_catalog_entry("nosuch").is_none());
        assert!(find_catalog_entry("").is_none());
    }

    #[test]
    fn entries_are_well_formed() {
        assert_eq!(CATALOG[0].name, "gemma-3-1b-q6k", "the recommended model comes first");
        let mut names = HashSet::new();
        let mut files = HashSet::new();
        for e in CATALOG {
            assert!(names.insert(e.name), "{} is listed twice", e.name);
            assert!(files.insert(e.file), "{} is listed twice", e.file);
            assert!(e.file.ends_with(".gguf"), "{}", e.file);
            assert!(
                e.url.starts_with("https://huggingface.co/") && e.url.ends_with(&format!("/{}", e.file)),
                "{}",
                e.url
            );
            assert!(e.bytes > 0 && !e.description.is_empty());
        }
    }
}
