//! nncpu runs large language models on the CPU: GGUF models of the Gemma 3
//! and 4, Llama 3 and Qwen 2, 2.5, 3, 3.5, 3.6 and 3.8 families (including
//! FunctionGemma, TranslateGemma and Gemma 4's mixture of experts), the
//! EmbeddingGemma text-embedding models, and speculative decoding with Gemma
//! 4's multi-token-prediction drafters. No GPU, no Python, no ML framework:
//! weights are decoded at load into 8-bit (or 4-bit, or BF16) layouts and run
//! through AVX2/FMA kernels, with portable scalar kernels on other CPUs.
//!
//! The [list of models known to work](https://github.com/ZacWalk/nncpu/blob/main/docs/models.md)
//! links to a download for each one.
//!
//! ```no_run
//! use nncpu::{GenerateParams, Model, Piece, Session, SessionParams};
//! use std::io::Write;
//!
//! fn main() -> nncpu::Result<()> {
//!     let model = Model::load("models/gemma-3-1b-it-Q6_K.gguf")?;
//!     let params = SessionParams { system_prompt: "Answer in two sentences.".into(), ..Default::default() };
//!     let mut chat = Session::new(&model, &params)?;
//!     let print = |piece: Piece| {
//!         if let Piece::Text(text) = piece {
//!             print!("{text}");
//!             std::io::stdout().flush().ok();
//!         }
//!         true // false would stop the reply
//!     };
//!     chat.chat("Why is the sky blue?", &GenerateParams::default(), print)?;
//!     println!();
//!     // The session remembers the conversation.
//!     chat.chat("Then why are sunsets red?", &GenerateParams::default(), print)?;
//!     println!();
//!     Ok(())
//! }
//! ```
//!
//! Threading: a [`Model`] is immutable once loaded, cheap to clone, and can
//! be shared by any number of sessions on any threads. A [`Session`] is used
//! by one thread at a time; [`Session::cancel_handle`] stops it from another.
//! All sessions share one process-wide worker pool (see [`set_num_threads`]),
//! so two sessions running at once is safe but not faster than running them
//! one after the other.

mod cpu;
#[doc(hidden)]
pub mod debug;
mod error;
pub mod gguf;
mod kernels;
mod model;
mod perf;
mod pool;
mod runtime;
mod session;
mod tokenizer;

pub use cpu::{CpuFeatures, cpu_features, cpu_supported};
pub use error::{Error, ErrorKind, Result};
pub use model::WeightFormat;
pub use perf::{perf_enable, perf_report, perf_reset};
pub use pool::{num_threads, set_num_threads};
pub use session::{
    CancelHandle, GenerateParams, GenerateResult, KvType, Piece, SamplerParams, Session, SessionParams, StopReason,
    Token,
};

use model::{ModelData, model_embed};
use std::path::Path;
use std::sync::Arc;

/// The library version, e.g. "0.1.0".
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// How a model is loaded.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ModelParams {
    /// How the weights are held in memory.
    pub weights: WeightFormat,
}

/// What a loaded model is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelInfo {
    /// general.name, or the file name.
    pub name: String,
    /// "gemma3", "gemma4", "llama", "qwen2", "qwen3", "qwen35",
    /// "gemma-embedding", ...
    pub architecture: String,
    pub weights: WeightFormat,
    pub n_vocab: usize,
    /// Context length the model was trained for.
    pub n_ctx_train: usize,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    /// Weights in the file.
    pub n_params: u64,
    /// Size of the GGUF file.
    pub file_bytes: u64,
    /// Weight bytes decoded into RAM (not counting the file mapping).
    pub resident_bytes: u64,
    /// [`Session::chat`] knows this model's turn format.
    pub has_chat_template: bool,
    /// Text-embedding models: values per [`Model::embed`] vector; 0 for
    /// generative models.
    pub n_embd_out: usize,
    /// TranslateGemma: made for [`Session::translate`].
    pub translator: bool,
    /// MTP drafters: the width (n_embd) of the model they draft for; else 0.
    pub n_embd_target: usize,
    /// It can reason before it answers: see [`SessionParams::thinking`].
    pub has_thinking: bool,
}

/// A loaded GGUF model. Cloning is cheap: clones share the weights.
#[derive(Clone)]
pub struct Model(pub(crate) Arc<ModelData>);

impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model")
            .field("name", &self.0.name)
            .field("arch", &self.0.hparams.arch)
            .finish()
    }
}

impl Model {
    /// Load a GGUF model with the default parameters (Q8_0 weights).
    pub fn load(path: impl AsRef<Path>) -> Result<Model> {
        Model::load_with(path, ModelParams::default())
    }

    /// Load a GGUF model.
    pub fn load_with(path: impl AsRef<Path>, params: ModelParams) -> Result<Model> {
        Ok(Model(Arc::new(model::load(path.as_ref(), params.weights, None)?)))
    }

    /// Load a GGUF model, reporting progress in [0, 1] to `progress`, which
    /// returns false to cancel the load.
    pub fn load_with_progress(
        path: impl AsRef<Path>,
        params: ModelParams,
        mut progress: impl FnMut(f32) -> bool,
    ) -> Result<Model> {
        Ok(Model(Arc::new(model::load(
            path.as_ref(),
            params.weights,
            Some(&mut progress),
        )?)))
    }

    /// general.name, or the file name.
    pub fn name(&self) -> &str {
        &self.0.name
    }

    /// The path the model was loaded from.
    pub fn path(&self) -> &str {
        &self.0.path
    }

    pub fn info(&self) -> ModelInfo {
        let m = &*self.0;
        let h = &m.hparams;
        let has_chat_template =
            m.vocab.chat != tokenizer::ChatFormat::None && h.n_embd_out == 0 && h.n_embd_target == 0;
        ModelInfo {
            name: m.name.clone(),
            architecture: h.arch.clone(),
            weights: m.weights,
            n_vocab: h.n_vocab,
            n_ctx_train: h.n_ctx,
            n_embd: h.n_embd,
            n_layer: h.n_layer,
            n_head: h.n_head,
            n_head_kv: h.n_head_kv,
            n_params: m.n_params,
            file_bytes: m.gguf.as_ref().map_or(0, |g| g.file_size),
            resident_bytes: m.decoded_bytes as u64,
            has_chat_template,
            n_embd_out: h.n_embd_out,
            translator: !m.vocab.languages.is_empty(),
            n_embd_target: h.n_embd_target,
            has_thinking: has_chat_template && (m.vocab.can_think || m.vocab.always_thinks),
        }
    }

    /// Tokens in the vocabulary.
    pub fn n_vocab(&self) -> usize {
        self.0.hparams.n_vocab
    }

    /// The beginning-of-sequence token, if the model has one.
    pub fn token_bos(&self) -> Option<Token> {
        let id = self.0.vocab.bos_id;
        (id >= 0).then_some(id)
    }

    /// The end-of-sequence token, if the model has one.
    pub fn token_eos(&self) -> Option<Token> {
        let id = self.0.vocab.eos_id;
        (id >= 0).then_some(id)
    }

    /// True for tokens that end generation: end-of-sequence and end-of-turn.
    pub fn is_eog(&self, token: Token) -> bool {
        self.0.vocab.is_eog(token)
    }

    /// Tokenize UTF-8 text. Spellings of control tokens in the text
    /// ("<start_of_turn>") are kept as plain text; user-defined tokens (such
    /// as FunctionGemma's function tags) are matched whole.
    pub fn tokenize(&self, text: &str, add_bos: bool) -> Vec<Token> {
        self.0.vocab.tokenize(text, add_bos)
    }

    /// The bytes `token` renders to: byte tokens become the byte, control
    /// tokens become nothing. A piece may be part of a UTF-8 character,
    /// because some characters span several tokens. None for an invalid token.
    pub fn token_piece(&self, token: Token) -> Option<&[u8]> {
        self.0.vocab.valid(token).then(|| self.0.vocab.piece(token))
    }

    /// The vocabulary entry for `token` as the file spells it ("▁Paris",
    /// "<0x0A>", "<end_of_turn>"). Meant for debugging.
    pub fn token_text(&self, token: Token) -> Option<&str> {
        self.0
            .vocab
            .valid(token)
            .then(|| self.0.vocab.tokens[token as usize].as_str())
    }

    /// The text `tokens` render to, with invalid UTF-8 replaced by U+FFFD.
    pub fn detokenize(&self, tokens: &[Token]) -> String {
        String::from_utf8_lossy(&self.0.vocab.detokenize(tokens)).into_owned()
    }

    /// A text-embedding model ([`ModelInfo::n_embd_out`] > 0) turns text into
    /// a vector whose direction stands for its meaning: texts about the same
    /// thing get vectors that point the same way, the basis of semantic
    /// search, clustering and deduplication.
    ///
    /// Writes the embedding of `text` to `out`, whose length may be smaller
    /// than n_embd_out to keep only the first values (EmbeddingGemma is
    /// trained for 512, 256 and 128 as well as its full 768); the vector has
    /// unit length, so the dot product of two is their cosine similarity.
    /// Text beyond the model's context is ignored.
    ///
    /// EmbeddingGemma expects a prefix naming the task, embedded with the
    /// text: "task: search result | query: " for a search query, "title: none
    /// | text: " for the documents it is matched against, "task: sentence
    /// similarity | query: " for texts compared with each other, and
    /// "task: clustering | query: " (likewise "classification", "question
    /// answering", "fact checking" and "code retrieval").
    ///
    /// The model is only read, so any number of threads may embed with it at
    /// once; each call allocates its own working memory.
    pub fn embed(&self, text: &str, out: &mut [f32]) -> Result<()> {
        model_embed(&self.0, text, out)
    }

    /// The full-length embedding of `text`; see [`embed`](Self::embed).
    pub fn embedding(&self, text: &str) -> Result<Vec<f32>> {
        let mut v = vec![0.0; self.0.hparams.n_embd_out];
        self.embed(text, &mut v)?;
        Ok(v)
    }
}
