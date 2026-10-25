# Using the nncpu app

The `nncpu` command: installing it, choosing a model, its options and how to
get the best speed. [models.md](models.md) lists every model known to work,
with a download link for each, the memory it needs and how fast it replies.

## Installing

With [Rust](https://rustup.rs) 1.88 or newer:

```text
cargo install --git https://github.com/ZacWalk/nncpu nncpu-cli   # installs the `nncpu` command
nncpu                      # offers to download a model, then chats
```

Or from a clone of this repository:

```text
git clone https://github.com/ZacWalk/nncpu
cd nncpu
cargo run --release -p nncpu-cli
```

With no model on disk, nncpu shows its download catalogue and asks before
downloading anything. The first entry, Gemma 3 1B (1 GB), is the recommended
first model. Then type a message at the `>` prompt. Ctrl-C stops a reply,
`/reset` starts a new conversation and `/exit` quits. Downloads use the
system's `curl` (or `wget`).

## Supported models

- **Chat and completion:** Gemma 3 and 4 (including the 26B A4B mixture of
  experts, FunctionGemma and TranslateGemma), Llama 3, Qwen 2, 2.5 and 3, and
  Qwen 3.5, 3.6 and 3.8 (hybrids whose Gated DeltaNet layers carry a
  recurrent state).
- **Text embeddings:** EmbeddingGemma 1 and 2.
- **Speculative decoding** with Gemma 4's multi-token-prediction (MTP)
  drafters: faster replies with exactly the same text.
- **Reasoning** ("thinking") for Gemma 4 and Qwen 3 / 3.5 / 3.6 / 3.8, with
  the reasoning delivered apart from the answer.

GGUF files in F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q4_K, Q5_K or
Q6_K load, which covers the common Q4_K_M, Q6_K and Q8_0 downloads.
IQ-quantised and Q2_K/Q3_K files do not; `nncpu --inspect-model FILE` tells
you whether a file will load.

## Choosing a model

Pass any of these names to `-m` (nncpu offers to download it if needed), or
download one with `nncpu --download NAME`:

| name | download | default: RAM, speed | `--q4`: RAM, speed | notes |
| --- | --- | --- | --- | --- |
| `gemma-3-270m-q8` | 0.3 GB | 0.3 GB, 97 tok/s | 0.2 GB, 159 tok/s | tiny; fine for simple tasks |
| `gemma-3-1b-q6k` | 1.0 GB | 1.0 GB, 28 tok/s | 0.6 GB, 45 tok/s | **start here**: small, quick, capable |
| `llama-3.2-1b-q6k` | 1.0 GB | 1.2 GB, 23 tok/s | 0.7 GB, 38 tok/s | Meta's small model |
| `qwen3.5-0.8b-q8` | 0.8 GB | 0.7 GB, 35 tok/s | 0.4 GB, 52 tok/s | Qwen's small hybrid model |
| `gemma-4-e2b-q4km` | 3.1 GB | 4.6 GB, 13 tok/s | 2.7 GB, 21 tok/s | much smarter; faster still with its drafter |
| `gemma-3-4b-q4km` | 2.5 GB | 3.8 GB, 7.6 tok/s | 2.3 GB, 12 tok/s | |
| `qwen3-4b-2507-q4km` | 2.5 GB | 4.0 GB, 6.9 tok/s | 2.3 GB, 11 tok/s | |
| `qwen3.5-4b-q4km` | 2.7 GB | 4.2 GB, 6.0 tok/s | 2.5 GB, 10 tok/s | |
| `gemma-4-e4b-q4km` | 5.0 GB | 7.4 GB, 6.1 tok/s | 4.3 GB, 9.9 tok/s | |
| `gemma-4-26b-a4b-q4km` | 16.8 GB | ~27 GB | 14.7 GB, 9.2 tok/s | mixture of experts: a 26B model at 4B speed; use `--q4` |
| `qwen3.8-27b-q4km` | 19.0 GB | 27 GB, 1.1 tok/s | 16 GB, 1.8 tok/s | the strongest here, and slow; use `--q4` |
| `translategemma-4b-q4km` | 2.5 GB | 3.8 GB, 7.6 tok/s | 2.3 GB, 12 tok/s | translation: `--translate en:de` |
| `functiongemma-270m-q8` | 0.3 GB | 0.3 GB, 102 tok/s | 0.2 GB, 165 tok/s | function calling ([example](../crates/nncpu/examples/function_call.rs)) |
| `embeddinggemma-300m-q8` | 0.3 GB | 0.3 GB | | text embeddings, texts up to 2K tokens |
| `embeddinggemma-2-q8` | 0.3 GB | 0.3 GB | | text embeddings, texts up to 8K tokens |
| `mtp-gemma-4-e2b-q8`, `mtp-gemma-4-e4b-q8`, `mtp-gemma-4-26b-a4b-q8` | 0.1–0.5 GB | | | drafters, fetched with `--download` rather than run: the Gemma 4 model of the same name then replies 1.2–2× faster, with the same text |

- **Speed** is the reply speed the C++ library measured on a Ryzen 9 5900XT
  with dual-channel DDR4-2400; nncpu generates at the same speed. Generating
  a token reads every weight once, so speed follows your memory bandwidth
  more than your core count.
- **RAM** is the weights as nncpu holds them. nncpu re-encodes the weights
  when it loads them, to 8-bit by default, so a 4-bit download takes about
  1.5× its file size in RAM. `--q4` keeps it near the file size and is about
  1.6× faster, at some cost in quality.
- **The conversation's K/V cache** comes on top of the RAM figure, and `-v`
  prints its size. At the default 4096-token context it is, for example,
  about 33 MB for Gemma 3 1B and Gemma 4 E2B, 128 MB for Llama 3.2 1B and
  576 MB for Qwen3 4B. It grows with `-c`, apart from Gemma's sliding-window
  layers and Qwen 3.5's recurrent layers, which stay the same size.
- **Other GGUF files** of a supported architecture work too: pass the path
  to `-m`. [models.md](models.md) lists every file known to work,
  with download links, including a few that aren't in the catalogue above.
- nncpu looks for models in `./models`, `~/.lmstudio/models` (so LM Studio's
  downloads are found) and `~/models`. Downloads go to `./models` when that
  directory exists, and to `~/models` otherwise.

## Running

```text
nncpu                                       # choose a model found on disk (or download one), then chat
nncpu -m gemma-4-e2b-q4km                   # a catalogue name; offers to download it if needed
nncpu -m D:\models\some-model-Q4_K_M.gguf   # or any GGUF file
nncpu -m qwen3.5-4b-q4km -p "Write a haiku about memory bandwidth"   # one reply, to stdout
nncpu -m gemma-4-e4b-q4km --think -c 16384  # reason before answering: better at code and maths
nncpu -m translategemma-4b-q4km --translate en:de -p "Good morning!"
nncpu -m embeddinggemma-300m-q8             # type sentences, see how alike they are
nncpu -m embeddinggemma-300m-q8 --task document < docs.txt > vectors.jsonl
```

In the chat, Ctrl-C stops a reply. `/reset` starts a new conversation,
`/translate en:de` translates each message instead of answering it
(`/translate off` to chat again), `/help` lists the commands and `/exit`
quits. An embedding model gets its own mode instead: typed lines are
compared with the ones before, and piped lines print one JSON vector each.

The options you are most likely to want (`--help` lists them all):

| option | what it does |
| --- | --- |
| `-m FILE\|NAME` | the model: a `.gguf` path, or a name from the table above |
| `-p TEXT` | answer one message, print the reply to stdout and exit |
| `-sys TEXT` | instructions for the model, given before the first message |
| `-c N` | the context window in tokens (default: up to 4096) |
| `-n N` | the most tokens per reply (default: until the reply ends) |
| `--temp X`, `-s N` | sampling temperature (default 0.7; 0 always picks the likeliest token), and a seed for repeatable replies |
| `--q4`, `--bf16` | 4-bit weights (less RAM, faster, lower quality), or 16-bit weights (more RAM, slower) |
| `--kv-f32` | keep the K/V cache in F32 rather than BF16, at twice the memory |
| `--think` | let the model reason before it answers (Gemma 4; Qwen 3, 3.5, 3.6, 3.8). It is better at hard problems but writes many more tokens; the reasoning shows dimmed on stderr and takes up context, so a larger `-c` helps |
| `-t N` | worker threads (default: `$NNCPU_THREADS`, else up to 8) |
| `--no-draft` | don't use a Gemma 4 drafter (`--draft FILE` picks one explicitly) |
| `--perf`, `-v` | a timing breakdown after each reply; what the library is doing |
| `--download [NAME]`, `--list-models` | download a model; list the models nncpu can see |
| `-y` | download without asking |

For looking inside models there are `--inspect-model`, `--inspect-all`,
`--gguf-info`, `--gguf-stats`, `--gemma-info`, `--gemma-probe`,
`--gemma-forward`, `--gemma-tokenize`, `--gemma-prompt`, `--gemma-gen` and
`--perplexity`; `--help` gives their arguments.

## Getting the best speed

- **Generation is limited by memory bandwidth.** Each new token reads every
  weight once, so a smaller model or `--q4` is faster. Dual-channel memory
  matters more than core count, and more than about 8 threads rarely helps.
- **Gemma 4 has drafters.** With `mtp-gemma-4-e2b-q8` (or the E4B or 26B
  one) downloaded, nncpu picks it up automatically. Replies come 1.2–2×
  faster, with exactly the same text.
- **Prompts are cheaper than replies.** nncpu reads a prompt 16 tokens per
  pass over the weights, so a long pasted document costs far less per token
  than the reply.
- **Thinking is paid for at reply speed.** With `--think`, a model may write
  hundreds or thousands of tokens of reasoning before its answer, so turn it
  on for the questions that need it.
- **Long contexts are cheapest on Gemma.** Each reply token reads the K/V
  cache as well as the weights. The cache is BF16, and Gemma's
  sliding-window layers keep only their window, so Gemma 3 1B holds an 8K
  context (`-c 8192`) in under 50 MB.
- **Build with `--release`.** A debug build is many times slower.
- `--perf` shows where the time goes, phase by phase, with the bandwidth each
  phase reached.
