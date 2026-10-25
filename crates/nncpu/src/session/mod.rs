//! Sessions: one conversation's context, K/V cache and sampler.
//!
//! The context is a token list whose first `cache.cur_pos` tokens have been
//! run through the model. The rest are "pending": appended but not yet
//! evaluated. Generation leaves its last token pending rather than paying a
//! forward pass that may never be needed; the next call evaluates it first.
//! `hidden` holds the model's final hidden state after the last evaluated
//! token, from which the next token's logits are computed on demand.

mod chat;
mod engine;
mod sample;

pub use chat::Piece;
pub use engine::Session;

use crate::Model;
use crate::error::{Result, bail};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// A token id.
pub type Token = i32;

/// Element type of a session's K/V cache.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum KvType {
    /// Half the memory, and half the bandwidth attention spends reading it.
    #[default]
    Bf16,
    /// Keys and values exactly as computed, at twice the memory.
    F32,
}

/// How a [`Session`] is set up.
#[derive(Clone, Default)]
pub struct SessionParams {
    /// Context window in tokens, at most `i32::MAX`; 0 = min(4096, the
    /// model's training context).
    pub n_ctx: usize,
    /// Sampling seed; 0 = random.
    pub seed: u64,
    /// Instructions for the model, given with the first chat turn.
    pub system_prompt: String,
    /// Optional speculative decoding with a multi-token-prediction (MTP)
    /// drafter: a small model made for this one (Gemma 4's "assistant"
    /// GGUFs, e.g. mtp-gemma-4-E2B-it-Q8_0.gguf for Gemma 4 E2B). It guesses
    /// the next few tokens and the model checks them all in one pass, which
    /// makes generation faster without changing what is generated: with Q8_0
    /// or Q4 weights and the same seed, the text is exactly the same as
    /// without it. Creating the session fails if it was not made for the
    /// model.
    pub draft_model: Option<Model>,
    /// Most tokens drafted per step; 0 = the default (4; 2 for
    /// mixture-of-experts models).
    pub n_draft: usize,
    /// Element type of the K/V cache.
    pub kv_type: KvType,
    /// Let the model reason before it answers, in every chat and translation
    /// turn, on models that can ([`ModelInfo::has_thinking`](crate::ModelInfo)):
    /// Gemma 4, and Qwen models whose chat template has a thinking switch
    /// (Qwen3, 3.5, 3.6 and 3.8). Off by default: it helps with hard problems
    /// such as code and maths, and makes every reply much longer. The
    /// reasoning arrives as [`Piece::Thinking`], never as [`Piece::Text`]. It
    /// stays in the context, where it counts towards `n_ctx`, so give a
    /// thinking session a larger context. Models made to always reason
    /// reason anyway.
    pub thinking: bool,
}

impl std::fmt::Debug for SessionParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionParams")
            .field("n_ctx", &self.n_ctx)
            .field("seed", &self.seed)
            .field("system_prompt", &self.system_prompt)
            .field("draft_model", &self.draft_model.as_ref().map(|m| m.name().to_string()))
            .field("n_draft", &self.n_draft)
            .field("kv_type", &self.kv_type)
            .field("thinking", &self.thinking)
            .finish()
    }
}

/// How the next token is chosen: top-k, then top-p, then min-p on the
/// model's own distribution, then temperature over what survives (the order
/// llama.cpp's default chain uses, so settings carry over).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplerParams {
    /// 0 = greedy: always the single most likely token.
    pub temperature: f32,
    /// Keep the k most likely tokens; 0 = all.
    pub top_k: usize,
    /// In [0, 1]: keep the fewest tokens reaching top_p; 1 = off.
    pub top_p: f32,
    /// In [0, 1]: drop tokens below min_p * p(most likely); 0 = off.
    pub min_p: f32,
}

impl Default for SamplerParams {
    /// Temperature 0.7, top_k 40, top_p 0.95, min_p 0.05.
    fn default() -> Self {
        SamplerParams {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
        }
    }
}

impl SamplerParams {
    /// Always the most likely token.
    pub fn greedy() -> Self {
        SamplerParams {
            temperature: 0.0,
            ..SamplerParams::default()
        }
    }

    pub(crate) fn check(&self, function: &str) -> Result<()> {
        let unit = |v: f32| v.is_finite() && (0.0..=1.0).contains(&v);
        if !(self.temperature.is_finite() && self.temperature >= 0.0 && unit(self.top_p) && unit(self.min_p)) {
            bail!(
                InvalidArgument,
                "{function}: temperature must be >= 0, and top_p/min_p in [0, 1]"
            );
        }
        Ok(())
    }
}

/// How a reply is generated.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GenerateParams {
    /// The most tokens to generate, reasoning included; 0 = until the end of
    /// the turn or a full context.
    pub max_tokens: usize,
    pub sampler: SamplerParams,
}

/// Why generation stopped.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum StopReason {
    /// The model produced an end-of-generation token.
    #[default]
    EndOfTurn,
    /// `max_tokens` were generated.
    MaxTokens,
    /// The context window filled up.
    ContextFull,
    /// The callback returned false.
    Callback,
    /// [`CancelHandle::cancel`] was called.
    Cancelled,
}

/// What a generation did.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GenerateResult {
    pub stop: StopReason,
    /// Tokens run through the model before generating.
    pub n_prompt: usize,
    /// Tokens generated, not counting the end-of-turn token.
    pub n_generated: usize,
    /// Time spent on the prompt.
    pub prefill: Duration,
    /// Time spent generating.
    pub decode: Duration,
    /// Tokens an MTP drafter proposed (0 without one).
    pub n_drafted: usize,
    /// ...and how many of them the model kept.
    pub n_draft_accepted: usize,
    /// Of n_generated, the tokens of reasoning, its markers included.
    pub n_thinking: usize,
}

/// Asks the call running on a session to stop at the next token boundary.
/// Cheap to clone and usable from any thread (also from a Ctrl-C handler).
/// Every call starts uncancelled, so a cancel that arrives while no call is
/// running is ignored.
#[derive(Clone, Debug)]
pub struct CancelHandle(Arc<AtomicBool>);

impl CancelHandle {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }
}
