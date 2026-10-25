# nncpu

Run large language models on your CPU, from Rust: no GPU, no Python, no ML
framework, and nothing you type leaves your machine. nncpu replaces libnncpu,
the C++ library it was ported from.

- **Chat and completion:** Gemma 3 and 4 (including the 26B A4B mixture of
  experts, FunctionGemma and TranslateGemma), Llama 3, and Qwen 2, 2.5, 3,
  3.5, 3.6 and 3.8.
- **Text embeddings:** EmbeddingGemma 1 and 2.
- **Speculative decoding** with Gemma 4's drafters (faster, same text), and
  **reasoning** delivered apart from the answer.
- **Quick on an ordinary PC:** the weights are re-encoded to 8 or 4 bits as
  they load and streamed through AVX2/FMA kernels, so a 1B model writes
  20–45 tokens a second on a desktop. Other CPUs, ARM included, run portable
  kernels that are correct but many times slower; `nncpu::cpu_supported()`
  tells you which you get.

GGUF files in F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q4_K, Q5_K or
Q6_K load (the common Q4_K_M, Q6_K and Q8_0 downloads). A small model needs
under 1 GB of memory, and a capable 4B one 2–4 GB. The
[list of models known to work](https://github.com/ZacWalk/nncpu/blob/main/docs/models.md)
links to a download for each. To start, put
[gemma-3-1b-it-Q6_K.gguf](https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q6_K.gguf)
(1 GB) in a `models` folder; the example below loads it from there.

## Example

```rust,no_run
use nncpu::{GenerateParams, Model, Piece, Session, SessionParams};
use std::io::Write;

fn main() -> nncpu::Result<()> {
    let model = Model::load("models/gemma-3-1b-it-Q6_K.gguf")?;
    let params = SessionParams { system_prompt: "Answer in two sentences.".into(), ..Default::default() };
    let mut chat = Session::new(&model, &params)?;

    let print = |piece: Piece| {
        if let Piece::Text(text) = piece {
            print!("{text}");
            std::io::stdout().flush().ok();
        }
        true // false would stop the reply
    };
    chat.chat("Why is the sky blue?", &GenerateParams::default(), print)?;
    println!();
    // The session remembers the conversation.
    chat.chat("Then why are sunsets red?", &GenerateParams::default(), print)?;
    println!();
    Ok(())
}
```

More in [`examples/`](examples): `minimal.rs` (one chat message), `embed.rs`
(semantic search with an embedding model) and `function_call.rs` (a
FunctionGemma call and its result).

## The API in brief

- `Model`: `Model::load(path)` (8-bit weights), or `Model::load_with` /
  `load_with_progress` with `ModelParams` to choose 4-bit
  (`WeightFormat::Q4`: less memory, ~1.6x faster, lower quality) or BF16
  weights. `info()`, `tokenize()`, `detokenize()`, `embed()`.
- `Session`: one conversation with its context window and K/V cache.
  `chat()` and `translate()` stream the reply to a callback as `Piece::Text`
  (the answer) or `Piece::Thinking` (the reasoning), in whole UTF-8
  characters; `generate()`, `eval()`, `logits()` and `sample()` work at the
  token level.
- `SessionParams`: context size, seed, system prompt, an optional MTP
  drafter (`draft_model`), the K/V cache type, and `thinking`.
- `CancelHandle`: stop a reply from another thread (or a Ctrl-C handler).
- `set_num_threads()` (default: `$NNCPU_THREADS`, else up to 8), and a phase
  profiler (`perf_enable()` / `perf_report()`).

A model is immutable once loaded and cheap to clone, so any number of
sessions on any threads can share it; embedding needs no session at all.
Set `NNCPU_NO_SIMD=1` to force the portable kernels (a debugging aid).

## License

MIT
