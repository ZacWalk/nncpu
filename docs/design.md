# nncpu design

How nncpu runs a model, how it differs from the C++ library it was ported
from, and how it was checked against it.

## How it works

- **Loading.** The GGUF file is memory-mapped and parsed in place. Packed
  weights (F16, BF16, Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0 and Q4_K / Q5_K /
  Q6_K) are decoded once, in parallel, into nncpu's own layout: 8-bit with
  BF16 block scales by default, 4-bit under `--q4`, or BF16 under `--bf16`.
  F32 tensors, and BF16 ones under `--bf16`, are used in place.
- **Kernels.** The matrix-vector products, attention, softmax, the norms,
  RoPE, soft-capping, SwiGLU and GELU are written with `std::arch` AVX2/FMA
  intrinsics, chosen at run time when the CPU has them, with portable scalar
  versions for every other CPU. The gemv kernels keep four independent FMA
  accumulators, so FMA latency does not hold back the stream of weights.
- **Batched prefill.** One pass over the weights serves 16 prompt tokens,
  and the 8-bit and 4-bit kernels work through them three at a time, which
  makes prompt processing compute-bound rather than bandwidth-bound.
- **Threads.** A static worker pool (by default `min(8, CPUs)`) splits every
  large matrix by rows. It also splits attention across positions for
  prompts, embedded texts and long-context replies.
- **Compact context.** The K/V cache is BF16, and sliding-window layers keep
  a ring of their recent rows.
- **Speculative decoding.** A Gemma 4 model uses its `mtp-*.gguf` drafter to
  guess several tokens, then checks them all in one pass. With 8-bit or 4-bit
  weights, the output is exactly what it would be without the drafter.
- **No computation graph.** The forward pass calls the kernels directly, and
  nothing is allocated per token.

## Differences from the C++ library

- **No JIT.** The C++ library generates x86-64 machine code for each matrix
  shape at run time; nncpu uses compiled AVX2/FMA intrinsics instead, with
  the same data layouts and the same order of operations where it matters.
  Its 8-bit and 4-bit weights are bit-for-bit those of the C++ library.
- **Any CPU.** Where the C++ library stops on a CPU without AVX2 and FMA,
  nncpu falls back to portable scalar kernels, so it also runs (slowly) on
  ARM. `NNCPU_NO_SIMD=1` forces them, as a debugging aid.
- **A Rust API** in place of the C API: `Model` and `Session` types, closures
  for streaming, `Result` for errors and the `log` crate for logging.
- **The app** reads `NNCPU_THREADS` (for `LLMCPU_THREADS`), has no `--test`
  mode (`cargo test` runs the tests) and downloads with the system's `curl`
  or `wget` on every platform.

## Validation

nncpu was checked against the C++ library on real models:

- **Logits.** Fed the same tokens with an F32 K/V cache, nncpu's logits
  differ from the C++ library's by a relative RMS of about 1e-6, with the
  same top token at every position, for Gemma 3 270M, 1B and 4B,
  FunctionGemma, Gemma 4 E2B and E4B, Gemma 4 26B A4B (`--q4`), Llama 3.2 1B,
  Llama 3 8B and Qwen 3.5 0.8B, with Q8_0, Q4 and BF16 weights. With the
  default BF16 cache, tiny rounding differences can flip the rounding of
  cached values, so the logits differ by about 1e-3, as they would between
  two C++ compilers.
- **Replies.** The app's output is identical to the C++ app's for greedy and
  seeded chats, translation, function calling, thinking, and Gemma 4 with
  its MTP drafter.
- **Embeddings** have a cosine similarity of at least 0.9999995 with the C++
  library's, and **perplexity** agrees to within 0.03%.
- **Speed.** Generation runs at the same speed as the C++ library; prompt
  processing is about 5–10% slower.

## Building and testing

```text
cargo build --release                  # the library and the app: target/release/nncpu
cargo test --workspace --release       # the tests need no model files
NNCPU_NO_SIMD=1 cargo test --workspace --release   # the same, on the portable kernels
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check
```

CI runs these on Linux, Windows and macOS, and builds with the minimum
supported Rust version, 1.88. The Model check workflow, started by hand from
the Actions tab, runs a real model on ARM runners (macOS on Apple silicon,
Linux and Windows) and on x64 ones, and compares their replies and speed.

The animation at the top of the README is recorded from a real session by
[nncpu-demo.py](nncpu-demo.py); its docstring explains how to remake it.
