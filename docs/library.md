# Using the nncpu library

Add it to your `Cargo.toml`:

```toml
[dependencies]
nncpu = "0.1"
```

A complete program:

```rust,no_run
use nncpu::{GenerateParams, Model, Piece, Session, SessionParams};
use std::io::Write;

fn main() -> nncpu::Result<()> {
    let model = Model::load("models/gemma-3-1b-it-Q6_K.gguf")?;
    let params = SessionParams { system_prompt: "Answer in two sentences.".into(), ..Default::default() };
    let mut chat = Session::new(&model, &params)?;

    // Receives the reply as it is generated, in whole UTF-8 characters.
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

[crates/nncpu/examples](../crates/nncpu/examples) has three complete programs:
[minimal.rs](../crates/nncpu/examples/minimal.rs) (one chat message),
[embed.rs](../crates/nncpu/examples/embed.rs) (semantic search with an embedding
model) and [function_call.rs](../crates/nncpu/examples/function_call.rs) (a
FunctionGemma call and its result):

```text
cargo run --release --example minimal -- models/gemma-3-1b-it-Q6_K.gguf "Why is the sky blue?"
```

## The API at a glance

| items | for |
| --- | --- |
| `Model::load`, `Model::load_with`, `Model::load_with_progress`, `ModelParams`, `Model::info` | loading a GGUF file, with 8-bit (default), 4-bit or BF16 weights |
| `Session::new`, `SessionParams`, `Session::reset` | a conversation: context size, seed, system prompt, drafter, K/V type and thinking |
| `Session::chat` | one chat turn in the model's own template: the answer streams as `Piece::Text` and, with thinking on, the reasoning as `Piece::Thinking` |
| `Session::translate` | one translation turn (TranslateGemma, or any chat model) |
| `Session::generate`, `Session::eval`, `Session::logits`, `Session::sample` | working with raw tokens: completion, scoring, your own sampling |
| `Model::tokenize`, `Model::token_piece`, `Model::detokenize`, `Model::is_eog` | the tokenizer |
| `Model::embed`, `Model::embedding` | a unit-length embedding vector from an embedding model; no session needed |
| `CancelHandle` | stopping a running call from another thread or a Ctrl-C handler |
| `set_num_threads`, `perf_enable`, `perf_report`, `cpu_supported` | process-wide settings, the phase profiler and the CPU check |

The rules that matter:

- A `Model` is immutable once loaded and cheap to clone (it is reference
  counted). Any number of sessions on any threads can share it, and
  embedding needs no session at all.
- A `Session` is driven by one thread at a time (`&mut self`); its
  `CancelHandle` can stop it from any other.
- Errors are `nncpu::Error`, with an `ErrorKind` to match on.
- Streamed text always arrives in whole UTF-8 characters.
- All sessions share one worker pool, so two sessions running at once is
  safe but not faster.
- Logging goes through the [`log`](https://crates.io/crates/log) crate.

The [crate documentation](https://docs.rs/nncpu) covers every item.
