//! The model: loader and forward pass for the Gemma 3, Gemma 4, Llama and
//! Qwen (2, 2.5, 3, 3.5, 3.6, 3.8) families, and the EmbeddingGemma
//! text-embedding models built on Gemma.
//!
//! [`load()`] maps a GGUF file, reads the hyperparameters and vocabulary, and
//! builds tensor views of every weight. F32/BF16 weights are used in place
//! from the mapping where the target layout allows; anything else is decoded
//! once, at load, into the requested in-memory format (Q8_0 by default). The
//! forward pass calls the kernels directly: there is no graph.

mod draft;
mod embed;
mod forward;
mod kv;
mod load;

pub(crate) use draft::{draft_init, draft_step};
pub(crate) use embed::{attn_span, model_embed};
pub(crate) use forward::{argmax, model_argmax, model_forward, model_hidden, model_logits, model_logits_batch};
pub(crate) use kv::{KvCache, kv_init, kv_rewind, state_restore, state_save};
pub(crate) use load::load;

#[cfg(test)]
pub(crate) mod tests;

use crate::gguf::GgufFile;
use crate::kernels::{self, RopeMode};
use crate::runtime::{AlignedBuf, DType, Tensor};
use crate::tokenizer::Vocab;

/// How a model's weights are held in memory. They are decoded into this
/// layout once, at load, whatever format the file stores them in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum WeightFormat {
    /// 8-bit, ~1.06 bytes/weight; near-lossless; the default.
    #[default]
    Q8_0,
    /// 4-bit, ~0.63 bytes/weight; ~1.6x faster, lower quality.
    Q4,
    /// 16-bit, used in place from the file where possible.
    Bf16,
}

impl WeightFormat {
    pub fn name(self) -> &'static str {
        match self {
            WeightFormat::Q8_0 => "Q8_0",
            WeightFormat::Q4 => "Q4",
            WeightFormat::Bf16 => "BF16",
        }
    }
}

/// The forward-pass family. EmbeddingGemma ("gemma-embedding") runs as gemma3
/// and EmbeddingGemma 2 ("gemma-embedding2") as gemma4, with bidirectional
/// attention and a pooled output instead of an lm_head.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Arch {
    #[default]
    Gemma3,
    Gemma4,
    Llama,
    /// Qwen2, Qwen2.5: Llama layers with Q/K/V biases.
    Qwen2,
    /// Qwen3: Llama layers with per-head Q/K norms.
    Qwen3,
    /// Qwen3.5, 3.6, 3.8: Gated DeltaNet layers, gated attention every fourth layer.
    Qwen35,
}

/// How an embedding model turns per-token hidden states into one vector
/// (GGUF `<arch>.pooling_type`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Pooling {
    /// Not an embedding model.
    #[default]
    None,
    Mean,
    /// The first token.
    Cls,
    Last,
}

/// Keys per attention block. Blocks are aligned to absolute positions, so a
/// query's result does not depend on how its work is shared out, and a
/// sliding layer's ring buffer is a whole number of blocks.
pub(crate) const ATTN_BLOCK: usize = 256;

/// Tokens per prefill batch. Large enough to amortise a weight pass over many
/// tokens, small enough that the batch's activations stay in L2.
pub(crate) const PREFILL_BATCH: usize = 16;

/// The most tokens [`model_embed`] looks at; longer input is cut.
pub(crate) const EMBED_MAX_TOKENS: usize = 8192;

/// Largest `cols / 32` whose Q4 block sums a single-vector gemv keeps on the
/// stack (cols <= 32768); wider matrices use the heap.
const MAX_XSUM_BLOCKS: usize = 1024;

/// Most experts per layer; bounds the bookkeeping of a mixture of experts.
pub(crate) const MAX_EXPERTS: usize = 512;

#[derive(Default)]
pub(crate) struct Layer {
    // Pre-norms (RMSNorm gamma, F32 vectors).
    pub attn_norm: Option<Tensor>,
    pub ffn_norm: Option<Tensor>,
    pub post_attention_norm: Option<Tensor>,
    pub post_ffw_norm: Option<Tensor>,
    pub post_norm: Option<Tensor>,
    pub attn_q_norm: Option<Tensor>,
    pub attn_k_norm: Option<Tensor>,

    // Attention projections.
    /// [n_embd, n_head*head_dim]; [q | gate] per head when attn_gated.
    pub attn_q: Option<Tensor>,
    pub attn_k: Option<Tensor>,
    /// Absent: V is K.
    pub attn_v: Option<Tensor>,
    pub attn_output: Option<Tensor>,
    /// Projection biases (Qwen2).
    pub attn_q_b: Option<Tensor>,
    pub attn_k_b: Option<Tensor>,
    pub attn_v_b: Option<Tensor>,

    // Gated DeltaNet (Qwen3.5's linear-attention layers), in place of
    // attention. Shapes are in Hparams (ssm_*).
    /// attn_qkv [n_embd, 2*n_k*d_k + n_v*d_v]
    pub ssm_qkv: Option<Tensor>,
    /// attn_gate [n_embd, n_v*d_v]: the output gate z.
    pub ssm_gate: Option<Tensor>,
    /// [n_embd, n_v] (F32): the decay's input.
    pub ssm_alpha: Option<Tensor>,
    /// [n_embd, n_v] (F32): the write strength's input.
    pub ssm_beta: Option<Tensor>,
    /// [d_conv, conv channels] (F32).
    pub ssm_conv1d: Option<Tensor>,
    /// [n_v] decay bias.
    pub ssm_dt: Option<Tensor>,
    /// [n_v] -exp(A_log): the decay rate.
    pub ssm_a: Option<Tensor>,
    /// [d_v] output norm gamma.
    pub ssm_norm: Option<Tensor>,
    pub ssm_out: Option<Tensor>,
    /// ssm_conv1d transposed: [d_conv][conv channels].
    pub ssm_conv_taps: Vec<f32>,

    // MLP projections.
    pub ffn_gate: Option<Tensor>,
    pub ffn_up: Option<Tensor>,
    pub ffn_down: Option<Tensor>,

    // Per-layer-input embedding gate (Gemma 4 PLE).
    pub inp_gate: Option<Tensor>,
    pub proj: Option<Tensor>,
    pub layer_output_scale: Option<Tensor>,

    // Mixture of experts (Gemma 4 26B A4B): beside the dense MLP, a router
    // picks n_expert_used of n_expert small gated MLPs per token. Expert
    // weights are stacked [cols, rows, n_expert] and stored as one
    // split-layout matrix per expert, back to back (see expert_matrix).
    /// [n_embd, n_expert] router (F32).
    pub ffn_gate_inp: Option<Tensor>,
    pub ffn_gate_inp_scale: Option<Tensor>,
    /// [n_embd, 2*n_ff_exp, n_expert]: gate rows, then up rows.
    pub ffn_gate_up_exps: Option<Tensor>,
    pub ffn_gate_exps: Option<Tensor>,
    pub ffn_up_exps: Option<Tensor>,
    pub ffn_down_exps: Option<Tensor>,
    /// [n_expert] per-expert output scale; optional.
    pub ffn_down_exps_scale: Option<Tensor>,
    pub pre_ffw_norm_2: Option<Tensor>,
    pub post_ffw_norm_1: Option<Tensor>,
    pub post_ffw_norm_2: Option<Tensor>,
    /// 0: no experts in this layer.
    pub n_ff_exp: usize,

    // Per-layer shapes, resolved once at load.
    pub n_ff: usize,
    pub head_dim: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub n_rot: usize,
    /// The layer whose K/V cache this layer reads: itself unless shared;
    /// None for Gated DeltaNet layers and drafter layers.
    pub kv_layer: Option<usize>,
    /// A Gated DeltaNet layer's slot in the recurrent state.
    pub rs_index: Option<usize>,
    /// Qwen3.5: attention output * sigmoid(gate), gate from attn_q.
    pub attn_gated: bool,
    pub sliding_window: bool,
    /// RoPE frequency divisors for this layer (n_rot/2 floats).
    pub rope_freqs: Option<Tensor>,
}

#[derive(Clone, Debug)]
pub(crate) struct Hparams {
    /// general.architecture: "gemma3", "gemma4", "llama", "qwen35", ...
    pub arch: String,
    pub arch_id: Arch,
    pub n_vocab: usize,
    /// `<arch>.context_length` (the training context).
    pub n_ctx: usize,
    pub n_embd: usize,
    /// Layers run; a Qwen3.5 file's trailing MTP block is not one.
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub head_dim_swa: usize,
    pub rope_dim: usize,
    pub rope_dim_swa: usize,
    pub rope_freq_base: f32,
    pub rope_freq_base_swa: f32,
    /// 1/rope.scaling.factor (linear scaling).
    pub rope_freq_scale: f32,
    pub rope_mode: RopeMode,
    pub rms_eps: f32,
    pub sliding_window: usize,
    /// Trailing layers that reuse an earlier layer's KV.
    pub shared_kv_layers: usize,
    /// Layers [0, n_layer_kv) own a KV cache.
    pub n_layer_kv: usize,
    /// embedding_length_per_layer_input (Gemma 4).
    pub ple_dim: usize,
    /// Experts per MoE layer (0: dense model).
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub final_logit_softcap: f32,
    pub attention_scale: f32,
    /// sqrt(n_embd) for Gemma, 1 for Llama and Qwen.
    pub embd_scale: f32,
    /// Raw GGUF general.file_type.
    pub file_type: i32,

    // Gated DeltaNet (qwen35): the causal convolution's width, the q/k heads
    // and their size, the value heads and their size, and how many layers
    // are Gated DeltaNet layers.
    pub ssm_d_conv: usize,
    pub ssm_n_k: usize,
    pub ssm_d_k: usize,
    pub ssm_n_v: usize,
    pub ssm_d_v: usize,
    pub n_layer_rs: usize,

    /// Text-embedding models attend both ways and pool their output into
    /// n_embd_out floats rather than predicting tokens.
    pub causal: bool,
    pub pooling: Pooling,
    /// 0: a generative model.
    pub n_embd_out: usize,

    /// MTP drafters (gemma4-assistant): the hidden width of the model they
    /// draft for; 0 for every other model.
    pub n_embd_target: usize,
}

impl Default for Hparams {
    fn default() -> Self {
        Hparams {
            arch: String::new(),
            arch_id: Arch::Gemma3,
            n_vocab: 0,
            n_ctx: 0,
            n_embd: 0,
            n_layer: 0,
            n_head: 0,
            n_head_kv: 0,
            head_dim: 0,
            head_dim_swa: 0,
            rope_dim: 0,
            rope_dim_swa: 0,
            rope_freq_base: 0.0,
            rope_freq_base_swa: 0.0,
            rope_freq_scale: 1.0,
            rope_mode: RopeMode::Neox,
            rms_eps: 0.0,
            sliding_window: 0,
            shared_kv_layers: 0,
            n_layer_kv: 0,
            ple_dim: 0,
            n_expert: 0,
            n_expert_used: 0,
            final_logit_softcap: 0.0,
            attention_scale: 1.0,
            embd_scale: 1.0,
            file_type: 0,
            ssm_d_conv: 0,
            ssm_n_k: 0,
            ssm_d_k: 0,
            ssm_n_v: 0,
            ssm_d_v: 0,
            n_layer_rs: 0,
            causal: true,
            pooling: Pooling::None,
            n_embd_out: 0,
            n_embd_target: 0,
        }
    }
}

/// A loaded model. Immutable after load; tensors point into `gguf`'s mapping
/// or into `buffers`, both owned here.
#[derive(Default)]
pub(crate) struct ModelData {
    pub path: String,
    /// general.name, else the file name.
    pub name: String,
    /// Owns the mapping (None only for synthetic test models).
    pub gguf: Option<GgufFile>,
    pub hparams: Hparams,
    pub vocab: Vocab,

    // Top-level weights.
    /// [n_embd, n_vocab]
    pub token_embd: Option<Tensor>,
    /// [ple_dim*n_layer, n_vocab]
    pub per_layer_token_embd: Option<Tensor>,
    /// [n_embd, ple_dim*n_layer]
    pub per_layer_model_proj: Option<Tensor>,
    /// [ple_dim] (F32)
    pub per_layer_proj_norm: Option<Tensor>,
    /// [n_embd] (F32)
    pub output_norm: Option<Tensor>,
    /// [n_embd, n_vocab]; absent = tied to token_embd.
    pub output: Option<Tensor>,
    /// [rope_dim/2] (F32)
    pub rope_freqs: Option<Tensor>,

    /// Embedding models: linear maps applied in order to the pooled hidden
    /// state (EmbeddingGemma: dense_2 then dense_3; EmbeddingGemma 2: its
    /// output projection). Empty for generative models.
    pub embd_proj: Vec<Tensor>,

    /// MTP drafters: [target embedding | target hidden] -> n_embd.
    pub nextn_pre: Option<Tensor>,
    /// MTP drafters: n_embd -> the next step's hidden.
    pub nextn_post: Option<Tensor>,

    pub layers: Vec<Layer>,

    /// Loader-owned weight buffers: decoded packed tensors and quantized
    /// copies. Never read: it owns the memory that the tensors point into.
    #[expect(
        dead_code,
        reason = "keeps the decoded weights alive for the tensors that point into them"
    )]
    pub buffers: Vec<AlignedBuf>,

    pub weights: WeightFormat,
    /// Total size of `buffers` (and other decoded data).
    pub decoded_bytes: usize,
    pub n_params: u64,
}

impl std::fmt::Debug for ModelData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelData")
            .field("name", &self.name)
            .field("arch", &self.hparams.arch)
            .finish()
    }
}

impl ModelData {
    /// The lm_head: `output`, or the tied token embeddings.
    pub fn lm_head(&self) -> &Tensor {
        self.output
            .as_ref()
            .or(self.token_embd.as_ref())
            .expect("a loaded model has token_embd")
    }

    pub fn token_embd(&self) -> &Tensor {
        self.token_embd.as_ref().expect("a loaded model has token_embd")
    }

    pub fn is_recurrent(&self) -> bool {
        self.hparams.n_layer_rs > 0
    }
}

/// An F32 vector's values.
#[inline]
pub(crate) fn f32s(t: &Option<Tensor>) -> &[f32] {
    t.as_ref().expect("a tensor checked at load").f32s()
}

/// Bytes of weight storage a gemv over `t` streams (one expert of a stack).
/// Used only for the perf report's effective-bandwidth column.
pub(crate) fn weight_bytes(t: &Tensor) -> u64 {
    let n = t.ne[0] as u64 * t.ne[1] as u64;
    match t.dtype {
        DType::Q8_0 => n + n / 32 * 2,
        DType::Q4S => n / 2 + n / 32 * 4,
        _ => n * 2,
    }
}

pub(crate) fn opt_weight_bytes(t: &Option<Tensor>) -> u64 {
    t.as_ref().map_or(0, weight_bytes)
}

/// Expert `e` of a stacked [cols, rows, n_expert] tensor as the 2-D matrix
/// the gemv paths take: the loader stores each expert as a complete
/// split-layout matrix, weight_bytes() long, back to back.
pub(crate) fn expert_matrix(t: &Tensor, e: usize) -> Tensor {
    let mut v = t.clone();
    v.n_dims = 2;
    v.ne[2] = 1;
    // SAFETY: e < n_expert, and the stack holds n_expert such matrices.
    v.data = unsafe { t.data.add(e * weight_bytes(t) as usize) };
    v
}

/// y = W x for one vector, dispatching on W's layout.
pub(crate) fn model_gemv(w: &Tensor, x: &[f32], y: &mut [f32]) {
    let (rows, cols) = (w.rows(), w.cols());
    let (x, y) = (&x[..cols], &mut y[..rows]);
    match w.dtype {
        DType::Q8_0 => {
            let (qs, scales) = w.q8();
            kernels::gemv_q8_0(qs, scales, x, y, rows, cols);
        }
        DType::Q4S => {
            // The per-block sums of x are shared by every row, so they are
            // computed once here rather than inside the row kernel. On the
            // stack, unless the matrix is wider than any supported model's.
            let nb = cols / 32;
            let mut stack = [0.0f32; MAX_XSUM_BLOCKS];
            let mut heap = Vec::new();
            let xsum = if nb <= MAX_XSUM_BLOCKS {
                &mut stack[..nb]
            } else {
                heap.resize(nb, 0.0);
                &mut heap[..]
            };
            kernels::block_sums(x, xsum);
            let (qs, scales, biases) = w.q4();
            kernels::gemv_q4(qs, scales, biases, x, xsum, y, rows, cols);
        }
        _ => kernels::gemv_bf16(w.bf16(), x, y, rows, cols),
    }
}

/// Row `token` of an embedding table, times `scale`, into `y` (y.len()
/// values).
pub(crate) fn model_embed_row(t: &Tensor, y: &mut [f32], token: usize, scale: f32) {
    match t.dtype {
        DType::Q8_0 => {
            let (qs, scales) = t.q8();
            kernels::embed_row_q8_0(y, qs, scales, token, scale);
        }
        DType::Q4S => {
            let (qs, scales, biases) = t.q4();
            kernels::embed_row_q4(y, qs, scales, biases, token, scale);
        }
        _ => kernels::embed_row_bf16(y, t.bf16(), token, scale),
    }
}
