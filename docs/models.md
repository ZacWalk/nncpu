# Models

These models are known to work with [nncpu](https://crates.io/crates/nncpu).
Each one is a single GGUF file on Hugging Face: download it (no account is
needed) and pass its path to `Model::load`, or to the `nncpu` app's `-m`.

Every file listed here was run with nncpu in October 2026 and gave exactly
the same replies as libnncpu, the C++ library nncpu was ported from.

**Where to start:** Gemma 3 1B is small, quick and capable. Gemma 4 E2B is
much smarter if you can spare about 5 GB of memory. For search and
similarity, use EmbeddingGemma.

## Chat

| Model | Download | Size | Memory, speed | 4-bit: memory, speed | Notes |
| --- | --- | --- | --- | --- | --- |
| [Gemma 3 270M](https://huggingface.co/ggml-org/gemma-3-270m-it-GGUF) | [gemma-3-270m-it-Q8_0.gguf](https://huggingface.co/ggml-org/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-Q8_0.gguf) | 0.3 GB | 0.3 GB, 97 tok/s | 0.2 GB, 159 tok/s | Tiny; fine for simple tasks |
| [Qwen2.5 0.5B Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF) | [qwen2.5-0.5b-instruct-q8_0.gguf](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf) | 0.7 GB | 0.6 GB, 56 tok/s | 0.4 GB, 91 tok/s | |
| [Qwen3 0.6B](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF) | [Qwen3-0.6B-Q8_0.gguf](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf) | 0.6 GB | 0.6 GB, 44 tok/s | 0.4 GB, 73 tok/s | Thinking |
| [Qwen3.5 0.8B](https://huggingface.co/ggml-org/Qwen3.5-0.8B-GGUF) | [Qwen3.5-0.8B-Q8_0.gguf](https://huggingface.co/ggml-org/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q8_0.gguf) | 0.8 GB | 0.7 GB, 35 tok/s | 0.4 GB, 52 tok/s | Thinking |
| [Gemma 3 1B](https://huggingface.co/unsloth/gemma-3-1b-it-GGUF) | [gemma-3-1b-it-Q6_K.gguf](https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q6_K.gguf) | 1.0 GB | 1.0 GB, 28 tok/s | 0.6 GB, 45 tok/s | **Start here**: small, quick, capable |
| [Llama 3.2 1B Instruct](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF) | [Llama-3.2-1B-Instruct-Q6_K.gguf](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q6_K.gguf) | 1.0 GB | 1.2 GB, 23 tok/s | 0.7 GB, 38 tok/s | Meta's small model |
| [Gemma 3 4B](https://huggingface.co/unsloth/gemma-3-4b-it-GGUF) | [gemma-3-4b-it-Q4_K_M.gguf](https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_M.gguf) | 2.5 GB | 3.8 GB, 7.6 tok/s | 2.3 GB, 12 tok/s | |
| [Qwen3 4B Instruct 2507](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF) | [Qwen3-4B-Instruct-2507-Q4_K_M.gguf](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf) | 2.5 GB | 4.0 GB, 6.9 tok/s | 2.3 GB, 11 tok/s | Answers without thinking |
| [Qwen3.5 4B](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF) | [Qwen3.5-4B-Q4_K_M.gguf](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf) | 2.7 GB | 4.2 GB, 6.0 tok/s | 2.5 GB, 10 tok/s | Thinking |
| [Gemma 4 E2B](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF) | [gemma-4-E2B-it-Q4_K_M.gguf](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf) | 3.1 GB | 4.6 GB, 13 tok/s | 2.7 GB, 21 tok/s | Much smarter; thinking; has a [drafter](#drafters) |
| [Gemma 4 E4B](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF) | [gemma-4-E4B-it-Q4_K_M.gguf](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf) | 5.0 GB | 7.4 GB, 6.1 tok/s | 4.3 GB, 9.9 tok/s | Thinking; has a [drafter](#drafters) |
| [Gemma 4 26B A4B](https://huggingface.co/lmstudio-community/gemma-4-26B-A4B-it-GGUF) | [gemma-4-26B-A4B-it-Q4_K_M.gguf](https://huggingface.co/lmstudio-community/gemma-4-26B-A4B-it-GGUF/resolve/main/gemma-4-26B-A4B-it-Q4_K_M.gguf) | 16.8 GB | ~27 GB | 14.7 GB, 9.2 tok/s | Mixture of experts: a 26B model at 4B speed. Use 4-bit weights; thinking; has a [drafter](#drafters) |
| [Qwen3.8 27B](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF) | [Qwen3.8-27B-Q4_K_M.gguf](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-Q4_K_M.gguf) | 19.0 GB | 27 GB, 1.1 tok/s | 16 GB, 1.8 tok/s | The strongest here, and slow. Use 4-bit weights; thinking |

## Translation and function calling

| Model | Download | Size | Memory, speed | 4-bit: memory, speed | Notes |
| --- | --- | --- | --- | --- | --- |
| [TranslateGemma 4B](https://huggingface.co/mradermacher/translategemma-4b-it-GGUF) | [translategemma-4b-it.Q4_K_M.gguf](https://huggingface.co/mradermacher/translategemma-4b-it-GGUF/resolve/main/translategemma-4b-it.Q4_K_M.gguf) | 2.5 GB | 3.8 GB, 7.6 tok/s | 2.3 GB, 12 tok/s | Translates between 55 languages with `Session::translate` |
| [FunctionGemma 270M](https://huggingface.co/ggml-org/functiongemma-270m-it-GGUF) | [functiongemma-270m-it-q8_0.gguf](https://huggingface.co/ggml-org/functiongemma-270m-it-GGUF/resolve/main/functiongemma-270m-it-q8_0.gguf) | 0.3 GB | 0.3 GB, 102 tok/s | 0.2 GB, 165 tok/s | Turns requests into function calls; see [function_call.rs](../crates/nncpu/examples/function_call.rs) |

Any chat model can translate with `Session::translate`, but TranslateGemma is
trained for it.

## Text embeddings

`Model::embedding` turns a text into a unit-length vector, with no session
needed.

| Model | Download | Size | Memory | Notes |
| --- | --- | --- | --- | --- |
| [EmbeddingGemma 300M](https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF) | [embeddinggemma-300M-Q8_0.gguf](https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF/resolve/main/embeddinggemma-300M-Q8_0.gguf) | 0.3 GB | 0.3 GB | 768 values; the first 512, 256 or 128 also work as a smaller vector (`Model::embed` with a shorter slice). Reads up to 2K tokens of text |
| [EmbeddingGemma 2](https://huggingface.co/ggml-org/embeddinggemma-2-GGUF) | [embeddinggemma-2-Q8_0.gguf](https://huggingface.co/ggml-org/embeddinggemma-2-GGUF/resolve/main/embeddinggemma-2-Q8_0.gguf) | 0.3 GB | 0.3 GB | 768 values. Reads up to 8K tokens of text |

## Drafters

A drafter makes a Gemma 4 model reply 1.2–2× faster, with exactly the same
text: it guesses several tokens ahead, and the model checks all the guesses
in one pass. Load the drafter with `Model::load` and pass it as
`SessionParams::draft_model`. The `nncpu` app uses one automatically when it
finds it beside the model or in its model folders.

| For | Download | Size |
| --- | --- | --- |
| Gemma 4 E2B | [mtp-gemma-4-E2B-it-Q8_0.gguf](https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF/resolve/main/mtp-gemma-4-E2B-it-Q8_0.gguf) | 0.1 GB |
| Gemma 4 E4B | [mtp-gemma-4-E4B-it-Q8_0.gguf](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/resolve/main/mtp-gemma-4-E4B-it-Q8_0.gguf) | 0.1 GB |
| Gemma 4 26B A4B | [mtp-gemma-4-26B-A4B-it-Q8_0.gguf](https://huggingface.co/ggml-org/gemma-4-26B-A4B-it-GGUF/resolve/main/mtp-gemma-4-26B-A4B-it-Q8_0.gguf) | 0.5 GB |

## Also known to work

- [gemma-3-1b-it-BF16.gguf](https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-BF16.gguf)
  (2.0 GB): Gemma 3 1B's original 16-bit weights, for
  `WeightFormat::Bf16`. With 8-bit or 4-bit weights it loads to the same
  model as the 6-bit file above.
- [Llama-3-8B-Instruct-Finance-RAG.Q4_K_S.gguf](https://huggingface.co/QuantFactory/Llama-3-8B-Instruct-Finance-RAG-GGUF/resolve/main/Llama-3-8B-Instruct-Finance-RAG.Q4_K_S.gguf)
  (4.7 GB), a fine-tune of Llama 3 8B: 8.1 GB of memory at 3.5 tok/s, or
  4.8 GB at 5.9 tok/s with 4-bit weights.

## Reading the tables

- **Memory** is what the weights take once loaded. nncpu re-encodes the
  weights as it loads them, to 8-bit by default. With 4-bit weights
  (`ModelParams { weights: WeightFormat::Q4 }`, or the app's `--q4`) a model
  takes less memory and replies about 1.6× faster, at some cost in quality.
- **A conversation's K/V cache** comes on top. At a 4096-token context it is,
  for example, about 33 MB for Gemma 3 1B and Gemma 4 E2B, 128 MB for
  Llama 3.2 1B and 576 MB for Qwen3 4B.
- **Speed** is how fast a reply is written, as measured for the C++ library
  on a Ryzen 9 5900XT with dual-channel DDR4-2400 and 8 threads; nncpu writes
  replies at the same speed. Each token reads every weight once, so speed
  follows memory bandwidth more than core count. Reading the prompt is much
  faster.
- **Thinking** models can reason before they answer: set
  `SessionParams::thinking` (the app's `--think`).
- **Licences:** each model has its own; see its model page.

## Other GGUF files

nncpu runs GGUF files of these architectures, fine-tunes included: `gemma3`,
`gemma4` (including its mixture of experts), `llama`, `qwen2`, `qwen3`,
`qwen35` (Qwen 3.5, 3.6 and 3.8), `gemma-embedding` and `gemma-embedding2`,
plus Gemma 4's drafters (`gemma4-assistant`).

The weights can be F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q4_K, Q5_K
or Q6_K, so the common Q4_K_M, Q4_K_S, Q6_K and Q8_0 files load. A file's
quantization only changes its size and load time, because nncpu re-encodes
the weights either way.

If a file can't run, `Model::load` returns an error that says why. The
`nncpu` app's `--inspect-model FILE` checks a file without running it.

These don't load:

| Files | Why |
| --- | --- |
| `IQ*`, `Q2_K` and `Q3_K` quantizations | nncpu can't decode them. Pick a Q4_K_M, Q6_K or Q8_0 file instead. |
| `mmproj-*.gguf` | Vision projectors: nncpu takes text only. |
| Qwen's mixtures of experts (`qwen2moe`, `qwen3moe`, `qwen35moe`, `qwen3next`) | Only Gemma 4's experts are implemented. This rules out Qwen3-30B-A3B, Qwen3.5/3.6-35B-A3B and Qwen3-Next. |
| Qwen3.8-Flash-Next (`qwen4exp`) | Its sparse attention, hyper-connections, n-gram embeddings and 512 experts aren't implemented. |
| Qwen3-Embedding and Qwen-VL | Qwen's embedding and vision models. |
| `mtp-Qwen*.gguf` and `dflash-*.gguf` | Drafters for other models or other drafting methods: only Gemma 4's `mtp-*` drafters work. |
