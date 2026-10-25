# nncpu

Run large language models on your CPU, in Rust. nncpu is a library you can
embed in your own programs and a command-line app for chatting with a model.
You don't need a graphics card, a cloud account or Python, and nothing you
type leaves your machine. It replaces libnncpu, the C++ library it was ported
from, and gives the same replies.

![nncpu chatting with Gemma 4 E2B in a terminal](https://raw.githubusercontent.com/ZacWalk/nncpu/main/docs/nncpu-demo.gif)

- **Models:** Gemma 3 and 4, Llama 3, Qwen 2, 2.5, 3, 3.5, 3.6 and 3.8, and
  EmbeddingGemma, from GGUF files ([list](docs/models.md)).
- **Quick on an ordinary PC:** weights are re-encoded to 8 or 4 bits and
  streamed through AVX2/FMA kernels; a 1B model writes 20–45 tokens a second
  on a desktop.
- **Runs anywhere:** portable kernels cover other CPUs, ARM included
  (correct, but many times slower).
- **More than chat:** reasoning, translation, function calling, embeddings
  and speculative decoding with Gemma 4's drafters.

## Quick start

With [Rust](https://rustup.rs) 1.88 or newer:

```text
cargo install --git https://github.com/ZacWalk/nncpu nncpu-cli
nncpu                      # offers to download a model, then chats
```

Gemma 3 1B (1 GB) is a good first model. Ctrl-C stops a reply and `/exit`
quits. [Using the app](docs/app.md) covers choosing a model, the options and
getting the best speed.

## Library

```toml
[dependencies]
nncpu = "0.1"
```

```rust,no_run
use nncpu::{GenerateParams, Model, Piece, Session, SessionParams};

fn main() -> nncpu::Result<()> {
    let model = Model::load("models/gemma-3-1b-it-Q6_K.gguf")?;
    let mut chat = Session::new(&model, &SessionParams::default())?;
    chat.chat("Why is the sky blue?", &GenerateParams::default(), |piece| {
        if let Piece::Text(text) = piece {
            print!("{text}");
        }
        true // false stops the reply
    })?;
    Ok(())
}
```

[Using the library](docs/library.md) has more examples and the API at a
glance, and [docs.rs](https://docs.rs/nncpu) documents every item.
[Design](docs/design.md) explains how nncpu works and how it was validated
against the C++ library.

## Development

```text
cargo test --workspace --release       # no model files needed
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check
```

nncpu provides the on-device AI in list-zero, a private desktop app.

## License

[MIT](LICENSE)
