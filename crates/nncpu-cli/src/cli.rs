//! Command-line options.
//!
//! One parser recognises every flag and mode; main() dispatches on the
//! result. Adding a flag means adding it here (parse + usage) and nowhere
//! else.

use crate::embed;
use crate::ui::fmt_g;
use nncpu::{SamplerParams, WeightFormat};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Mode {
    /// The default: pick or download a model, then chat (or answer -p once).
    #[default]
    Chat,
    Help,
    Version,
    ListModels,
    Download,
    GgufInfo,
    GgufStats,
    InspectModel,
    InspectAll,
    ModelInfo,
    Probe,
    Forward,
    Tokenize,
    Prompt,
    Gen,
    Perplexity,
}

#[derive(Clone, Debug, Default)]
pub struct Options {
    pub mode: Mode,
    /// The mode's positional arguments.
    pub args: Vec<String>,
    /// -m: a path, or a name from the download catalogue.
    pub model: String,
    /// -p: answer this one message and exit.
    pub prompt: String,
    /// --system
    pub system_prompt: String,
    /// --translate FROM:TO: translate instead of chatting.
    pub translate_from: String,
    pub translate_to: String,
    /// --task: embedding models' task prefix (embed.rs); "" = chosen by mode.
    pub task: String,
    /// --dim: embedding values to keep; 0 = all.
    pub dim: usize,
    /// --draft: an MTP drafter; "" = look for one (Gemma 4).
    pub draft: String,
    /// --no-draft
    pub no_draft: bool,
    /// --kv-f32: an F32 K/V cache.
    pub kv_f32: bool,
    /// --think: let the model reason before it answers.
    pub think: bool,
    /// --draft-n: most tokens drafted per step; 0 = library default.
    pub n_draft: usize,
    pub weights: WeightFormat,
    /// -n: most tokens per reply; 0 = until the turn ends.
    pub n_predict: usize,
    /// -c: 0 = min(4096, the model's training context).
    pub n_ctx: usize,
    /// -t: 0 = library default.
    pub n_threads: usize,
    pub sampler: SamplerParams,
    /// -s: 0 = random.
    pub seed: u64,
    pub perf: bool,
    pub verbose: bool,
    /// Answer yes to download questions.
    pub yes: bool,
}

/// The modes: flag, mode, fewest and most arguments.
const MODES: &[(&str, Mode, usize, usize)] = &[
    ("--help", Mode::Help, 0, 0),
    ("-h", Mode::Help, 0, 0),
    ("--version", Mode::Version, 0, 0),
    ("--list-models", Mode::ListModels, 0, 0),
    ("--download", Mode::Download, 0, 1),
    ("--gguf-info", Mode::GgufInfo, 1, 1),
    ("--gguf-stats", Mode::GgufStats, 1, 2),
    ("--inspect-model", Mode::InspectModel, 1, 1),
    ("--inspect-all", Mode::InspectAll, 0, 0),
    ("--gemma-info", Mode::ModelInfo, 1, 1),
    ("--gemma-probe", Mode::Probe, 1, 2),
    ("--gemma-forward", Mode::Forward, 1, 2),
    ("--gemma-tokenize", Mode::Tokenize, 2, 2),
    ("--gemma-prompt", Mode::Prompt, 2, 4),
    ("--gemma-gen", Mode::Gen, 2, 4),
    ("--perplexity", Mode::Perplexity, 2, 3),
];

/// Options that take a value, in all their spellings.
#[rustfmt::skip]
const VALUE_OPTIONS: &[&str] = &[
    "-m", "--model",
    "-p", "--prompt",
    "-sys", "--system",
    "-n", "--n_predict", "--n-predict",
    "-c", "--ctx-size", "--ctx_size",
    "-t", "--threads",
    "--temp",
    "--top_k", "--top-k",
    "--top_p", "--top-p",
    "--min_p", "--min-p",
    "-s", "--seed",
    "-b", "--batch_size",
    "--translate",
    "--task",
    "--dim",
    "--draft", "-md", "--model-draft",
    "--draft-n",
];

const FLAGS: &[&str] = &[
    "--bf16",
    "--q8",
    "--q4",
    "--perf",
    "-v",
    "--verbose",
    "-y",
    "--yes",
    "--no-draft",
    "--kv-f32",
    "--think",
];

fn find_mode(arg: &str) -> Option<&'static (&'static str, Mode, usize, usize)> {
    MODES.iter().find(|m| m.0 == arg)
}

fn known(arg: &str) -> bool {
    find_mode(arg).is_some() || VALUE_OPTIONS.contains(&arg) || FLAGS.contains(&arg)
}

/// Older spellings with one dash: "-bf16", "-temp" mean "--bf16", "--temp".
fn normalise(arg: &str) -> String {
    let b = arg.as_bytes();
    if b.len() > 2 && b[0] == b'-' && b[1] != b'-' && !known(arg) && known(&format!("-{arg}")) {
        format!("-{arg}")
    } else {
        arg.to_string()
    }
}

/// Parse the arguments after the program name. On failure, returns the
/// problem, e.g. "-t needs a whole number from 0 to 64, not '99'".
pub fn parse_options(argv: &[String]) -> Result<Options, String> {
    let mut o = Options::default();
    let mut have_mode = false;
    let mut i = 0;
    while i < argv.len() {
        let flag = argv[i].as_str();
        let arg = normalise(flag);
        i += 1;

        if let Some(&(_, mode, min_args, max_args)) = find_mode(&arg) {
            if have_mode && o.mode != mode {
                return Err(format!("{flag} cannot be combined with another mode"));
            }
            have_mode = true;
            o.mode = mode;
            // Required arguments are taken as they come; optional ones only
            // when they do not look like an option.
            let mut taken = 0;
            while taken < max_args && i < argv.len() && (taken < min_args || !argv[i].starts_with('-')) {
                o.args.push(argv[i].clone());
                i += 1;
                taken += 1;
            }
            if taken < min_args {
                let s = if min_args == 1 { "" } else { "s" };
                return Err(format!("{flag} needs {min_args} argument{s} (see --help)"));
            }
            continue;
        }

        if FLAGS.contains(&arg.as_str()) {
            match arg.as_str() {
                "--bf16" => o.weights = WeightFormat::Bf16,
                "--q8" => o.weights = WeightFormat::Q8_0,
                "--q4" => o.weights = WeightFormat::Q4,
                "--perf" => o.perf = true,
                "-v" | "--verbose" => o.verbose = true,
                "-y" | "--yes" => o.yes = true,
                "--no-draft" => o.no_draft = true,
                "--kv-f32" => o.kv_f32 = true,
                _ => o.think = true, // --think
            }
            continue;
        }

        if VALUE_OPTIONS.contains(&arg.as_str()) {
            let Some(v) = argv.get(i) else {
                return Err(format!("{flag} needs a value"));
            };
            i += 1;
            match arg.as_str() {
                "-m" | "--model" => o.model = v.clone(),
                "-p" | "--prompt" => o.prompt = v.clone(),
                "-sys" | "--system" => o.system_prompt = v.clone(),
                "-n" | "--n_predict" | "--n-predict" => o.n_predict = parse_int(flag, v, 0, i32::MAX as usize)?,
                "-c" | "--ctx-size" | "--ctx_size" => o.n_ctx = parse_int(flag, v, 0, 1 << 20)?,
                "-t" | "--threads" => o.n_threads = parse_int(flag, v, 0, 64)?,
                "--temp" => o.sampler.temperature = parse_float(flag, v, 0.0, 100.0)?,
                "--top_k" | "--top-k" => o.sampler.top_k = parse_int(flag, v, 0, i32::MAX as usize)?,
                "--top_p" | "--top-p" => o.sampler.top_p = parse_float(flag, v, 0.0, 1.0)?,
                "--min_p" | "--min-p" => o.sampler.min_p = parse_float(flag, v, 0.0, 1.0)?,
                "--translate" => {
                    let Some((from, to)) = parse_language_pair(v) else {
                        return Err(format!("{flag} needs two languages as FROM:TO, e.g. en:de, not '{v}'"));
                    };
                    (o.translate_from, o.translate_to) = (from, to);
                }
                "--task" => {
                    if embed::task_prefix(v).is_none() {
                        return Err(format!("{flag} needs one of {}, not '{v}'", embed::task_names()));
                    }
                    o.task = v.clone();
                }
                "--dim" => o.dim = parse_int(flag, v, 1, 1 << 16)?,
                "--draft" | "-md" | "--model-draft" => o.draft = v.clone(),
                "--draft-n" => o.n_draft = parse_int(flag, v, 1, 15)?,
                "-s" | "--seed" => {
                    // Negative (the old "-1") means "pick one at random", as does 0.
                    let Some(seed) = parse_long(v) else {
                        return Err(format!("{flag} needs a whole number, not '{v}'"));
                    };
                    o.seed = u64::try_from(seed).unwrap_or(0);
                }
                // -b is accepted for old scripts; prefill batching is fixed.
                _ => {}
            }
            continue;
        }

        return Err(format!("unknown argument '{flag}' (see --help)"));
    }
    Ok(o)
}

fn parse_int(flag: &str, s: &str, lo: usize, hi: usize) -> Result<usize, String> {
    parse_long(s)
        .and_then(|v| usize::try_from(v).ok())
        .filter(|v| (lo..=hi).contains(v))
        .ok_or_else(|| format!("{flag} needs a whole number from {lo} to {hi}, not '{s}'"))
}

fn parse_float(flag: &str, s: &str, lo: f32, hi: f32) -> Result<f32, String> {
    s.trim_start_matches(is_c_space)
        .parse::<f64>()
        .ok()
        .filter(|v| v.is_finite() && (f64::from(lo)..=f64::from(hi)).contains(v))
        .map(|v| v as f32)
        .ok_or_else(|| {
            format!(
                "{flag} needs a number from {} to {}, not '{s}'",
                fmt_g(lo.into(), 6),
                fmt_g(hi.into(), 6)
            )
        })
}

/// A whole number as C's strtol reads one, which must be all of `s`: leading
/// white space and a sign are allowed.
pub fn parse_long(s: &str) -> Option<i64> {
    let s = s.trim_start_matches(is_c_space);
    let digits = s.strip_prefix(['+', '-']).unwrap_or(s);
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    s.parse().ok()
}

fn is_c_space(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\x0B' | '\x0C' | '\r')
}

/// "en:de" -> ("en", "de"). None unless both sides are non-empty.
pub fn parse_language_pair(pair: &str) -> Option<(String, String)> {
    let (from, to) = pair.split_once(':')?;
    (!from.is_empty() && !to.is_empty() && !to.contains(':')).then(|| (from.to_string(), to.to_string()))
}

/// The --help text; `argv0` is the program as it was run.
pub fn usage(argv0: &str) -> String {
    format!(
        "nncpu {version} - run Gemma, Llama and Qwen GGUF models on the CPU

usage: {argv0} [options]            pick a model (or download one) and chat
       {argv0} -m MODEL -p TEXT     answer one message and exit

model:
  -m, --model FILE|NAME    a .gguf file, or a name from --download's list
                           (default: choose from the models found)
  --q8                     8-bit weights in memory (default)
  --q4                     4-bit weights: ~1.6x faster, lower quality
  --bf16                   keep 16-bit weights, used straight from the file
  -c, --ctx-size N         context window in tokens (default: up to 4096)
  --kv-f32                 keep the K/V cache in F32 (default BF16: half the memory)
  -t, --threads N          worker threads (default: $NNCPU_THREADS, else up to 8)
  --draft FILE             an MTP drafter, for faster replies (default: Gemma 4 models use
                           an mtp-<model>.gguf found beside them or in the model folders)
  --no-draft               do not use a drafter
  --draft-n N              most tokens the drafter guesses at a time (default 4; 2 for 26B A4B)

chat:
  -p, --prompt TEXT        answer TEXT and exit; the reply goes to stdout
  -sys, --system TEXT      instructions for the model, given before the first message
  --think                  let the model reason before it answers (Gemma 4; Qwen 3, 3.5, 3.6
                           and 3.8): better at hard problems, much longer replies; the
                           reasoning is shown dimmed on stderr and fills the context too
  -n, --n_predict N        most tokens per reply, reasoning included (default: until the
                           reply ends)
  --temp X                 sampling temperature; 0 = always the likeliest token (default 0.7)
  --top-k N                sample from the N likeliest tokens; 0 = all (default 40)
  --top-p X                ...that make up probability X (default 0.95)
  --min-p X                ...each at least X times the likeliest (default 0.05)
  -s, --seed N             random seed, for repeatable replies (default: random)
  --perf                   print a per-phase timing breakdown after each reply
  -v, --verbose            show what the library is doing
  -y, --yes                download without asking

  In the chat, /help lists commands; Ctrl-C stops a reply.

translation:
  --translate FROM:TO      translate each message instead of chatting, e.g. en:de
                           (made for TranslateGemma; works with any chat model)

embedding models (EmbeddingGemma):
  Typed lines are compared with the ones before; piped lines, or -p TEXT,
  print one JSON array of values per line.
  --task NAME              prefix for the task: {tasks}
                           (default: similarity when typing, none otherwise;
                           query and document for search)
  --dim N                  keep the first N values (EmbeddingGemma: 768, 512, 256 or 128)

modes (each runs instead of the chat):
  --download [NAME]        download a model (no NAME: choose from a list)
  --list-models            list the .gguf files nncpu finds
  --version                print the version
  --gguf-info FILE         a GGUF file's metadata and tensor table
  --gguf-stats FILE [NAME] statistics of the F32/F16/BF16 tensors whose name holds NAME
  --inspect-model FILE     whether nncpu can load FILE; --inspect-all checks every model found
  --gemma-info FILE, --gemma-probe FILE [ID], --gemma-forward FILE [ID],
  --gemma-tokenize FILE \"TEXT\", --gemma-prompt FILE \"TEXT\" [N] [CTX],
  --gemma-gen FILE \"ID ID ...\" [N] [CTX]
                           loader, tokenizer and forward-pass diagnostics (greedy)
  --perplexity FILE TEXTFILE [N]
                           how well the model predicts the first N tokens of TEXTFILE (default 512)

Models are looked for in ./models, ~/.lmstudio/models and ~/models.
",
        version = nncpu::version(),
        tasks = embed::task_names(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Options, String> {
        parse_options(&args.iter().map(|a| a.to_string()).collect::<Vec<_>>())
    }

    #[test]
    fn defaults() {
        let o = parse(&[]).unwrap();
        assert_eq!(o.mode, Mode::Chat);
        assert!(o.args.is_empty() && o.model.is_empty() && o.prompt.is_empty());
        assert_eq!(o.weights, WeightFormat::Q8_0);
        assert_eq!(
            o.sampler,
            SamplerParams {
                temperature: 0.7,
                top_k: 40,
                top_p: 0.95,
                min_p: 0.05
            }
        );
        assert_eq!(
            (o.n_predict, o.n_ctx, o.n_threads, o.seed, o.dim, o.n_draft),
            (0, 0, 0, 0, 0, 0)
        );
        assert!(!o.perf && !o.verbose && !o.yes && !o.think && !o.kv_f32 && !o.no_draft);
    }

    #[test]
    fn every_option() {
        let line = "-m model.gguf -p hello -sys brief -n 64 -c 2048 -t 4 --temp 0.2 --top-k 5 --top_p 0.5 --min-p 0.1 \
                    -s 42 --q4 -v -y --think --kv-f32 --no-draft --draft d.gguf --draft-n 3 --task query --dim 256 \
                    --translate pt-BR:en --perf -b 512";
        let o = parse(&line.split_whitespace().collect::<Vec<_>>()).unwrap();
        assert_eq!(o.mode, Mode::Chat);
        assert_eq!(
            (o.model.as_str(), o.prompt.as_str(), o.system_prompt.as_str()),
            ("model.gguf", "hello", "brief")
        );
        assert_eq!((o.n_predict, o.n_ctx, o.n_threads, o.seed), (64, 2048, 4, 42));
        assert_eq!(
            o.sampler,
            SamplerParams {
                temperature: 0.2,
                top_k: 5,
                top_p: 0.5,
                min_p: 0.1
            }
        );
        assert_eq!(o.weights, WeightFormat::Q4);
        assert!(o.verbose && o.yes && o.think && o.kv_f32 && o.no_draft && o.perf);
        assert_eq!((o.draft.as_str(), o.n_draft), ("d.gguf", 3));
        assert_eq!((o.task.as_str(), o.dim), ("query", 256));
        assert_eq!((o.translate_from.as_str(), o.translate_to.as_str()), ("pt-BR", "en"));
        assert_eq!(parse(&["--bf16"]).unwrap().weights, WeightFormat::Bf16);
        assert_eq!(parse(&["--q4", "--q8"]).unwrap().weights, WeightFormat::Q8_0);
        assert_eq!(parse(&["--model-draft", "x"]).unwrap().draft, "x");
        assert_eq!(parse(&["-md", "y"]).unwrap().draft, "y");
    }

    #[test]
    fn old_spellings() {
        let o = parse(&[
            "-temp",
            "0.5",
            "-bf16",
            "-top_k",
            "3",
            "-ctx_size",
            "100",
            "-n_predict",
            "7",
            "-batch_size",
            "8",
        ])
        .unwrap();
        assert_eq!(o.sampler.temperature, 0.5);
        assert_eq!(o.weights, WeightFormat::Bf16);
        assert_eq!((o.sampler.top_k, o.n_ctx, o.n_predict), (3, 100, 7));
        assert_eq!(parse(&["-version"]).unwrap().mode, Mode::Version);
        // "-sys" and "-md" are options of their own, not "--sys" and "--md".
        assert_eq!(parse(&["-sys", "x"]).unwrap().system_prompt, "x");
    }

    #[test]
    fn seeds() {
        assert_eq!(parse(&["-s", "-1"]).unwrap().seed, 0, "negative means random");
        assert_eq!(parse(&["--seed", "0"]).unwrap().seed, 0);
        assert_eq!(parse(&["--seed", "+7"]).unwrap().seed, 7);
        assert_eq!(parse(&["-s", "x"]).unwrap_err(), "-s needs a whole number, not 'x'");
    }

    #[test]
    fn modes_and_their_arguments() {
        let o = parse(&["--gguf-stats", "f.gguf", "blk.0", "-v"]).unwrap();
        assert_eq!((o.mode, o.verbose), (Mode::GgufStats, true));
        assert_eq!(o.args, ["f.gguf", "blk.0"]);
        // Optional arguments are not taken when they look like options.
        let o = parse(&["--gguf-stats", "f.gguf", "-v"]).unwrap();
        assert_eq!((o.args.len(), o.verbose), (1, true));
        // Required ones are taken as they come.
        let o = parse(&["--gemma-tokenize", "f.gguf", "-x"]).unwrap();
        assert_eq!(o.args, ["f.gguf", "-x"]);
        let o = parse(&["--gemma-prompt", "f.gguf", "Hello", "8", "128"]).unwrap();
        assert_eq!((o.mode, o.args.len()), (Mode::Prompt, 4));
        let o = parse(&["--download"]).unwrap();
        assert_eq!((o.mode, o.args.len()), (Mode::Download, 0));
        let o = parse(&["--download", "gemma-3-1b-q6k", "-y"]).unwrap();
        assert_eq!(o.args, ["gemma-3-1b-q6k"]);
        assert!(o.yes);
        let o = parse(&["--download", "a", "b"]);
        assert_eq!(o.unwrap_err(), "unknown argument 'b' (see --help)");
        assert_eq!(parse(&["-h"]).unwrap().mode, Mode::Help);
        assert_eq!(
            parse(&["--help", "-h"]).unwrap().mode,
            Mode::Help,
            "the same mode twice is fine"
        );
        assert_eq!(parse(&["--inspect-all"]).unwrap().mode, Mode::InspectAll);
        assert_eq!(parse(&["--perplexity", "m", "t.txt", "100"]).unwrap().args.len(), 3);
        let o = parse(&["--gemma-gen", "m", "2 106 1645"]).unwrap();
        assert_eq!((o.mode, o.args[1].as_str()), (Mode::Gen, "2 106 1645"));
        assert_eq!(parse(&["--gemma-probe", "m"]).unwrap().mode, Mode::Probe);
        assert_eq!(parse(&["--gemma-forward", "m", "5"]).unwrap().mode, Mode::Forward);
        assert_eq!(parse(&["--gemma-info", "m"]).unwrap().mode, Mode::ModelInfo);
        assert_eq!(parse(&["--inspect-model", "m"]).unwrap().mode, Mode::InspectModel);
        assert_eq!(parse(&["--gguf-info", "m"]).unwrap().mode, Mode::GgufInfo);
        assert_eq!(parse(&["--list-models"]).unwrap().mode, Mode::ListModels);
    }

    #[test]
    fn errors() {
        let err = |args: &[&str]| parse(args).unwrap_err();
        assert_eq!(
            err(&["--help", "--version"]),
            "--version cannot be combined with another mode"
        );
        assert_eq!(err(&["--gguf-info"]), "--gguf-info needs 1 argument (see --help)");
        assert_eq!(
            err(&["--gemma-tokenize", "m"]),
            "--gemma-tokenize needs 2 arguments (see --help)"
        );
        assert_eq!(err(&["-m"]), "-m needs a value");
        assert_eq!(
            err(&["--temp", "200"]),
            "--temp needs a number from 0 to 100, not '200'"
        );
        assert_eq!(
            err(&["--temp", "warm"]),
            "--temp needs a number from 0 to 100, not 'warm'"
        );
        assert_eq!(
            err(&["--top-p", "1.5"]),
            "--top-p needs a number from 0 to 1, not '1.5'"
        );
        assert_eq!(
            err(&["--min-p", "nan"]),
            "--min-p needs a number from 0 to 1, not 'nan'"
        );
        assert_eq!(err(&["-t", "65"]), "-t needs a whole number from 0 to 64, not '65'");
        assert_eq!(
            err(&["-n", "-1"]),
            "-n needs a whole number from 0 to 2147483647, not '-1'"
        );
        assert_eq!(
            err(&["-c", "2.5"]),
            "-c needs a whole number from 0 to 1048576, not '2.5'"
        );
        assert_eq!(
            err(&["--dim", "0"]),
            "--dim needs a whole number from 1 to 65536, not '0'"
        );
        assert_eq!(
            err(&["--draft-n", "16"]),
            "--draft-n needs a whole number from 1 to 15, not '16'"
        );
        assert_eq!(
            err(&["-temp", "x"]),
            "-temp needs a number from 0 to 100, not 'x'",
            "the flag as it was spelt"
        );
        assert_eq!(
            err(&["--translate", "en"]),
            "--translate needs two languages as FROM:TO, e.g. en:de, not 'en'"
        );
        assert_eq!(
            err(&["--task", "search"]),
            "--task needs one of none, query, document, similarity, classification, clustering, qa, fact, code, not \
             'search'"
        );
        assert_eq!(err(&["--bogus"]), "unknown argument '--bogus' (see --help)");
        assert_eq!(err(&["model.gguf"]), "unknown argument 'model.gguf' (see --help)");
        assert_eq!(err(&["--test"]), "unknown argument '--test' (see --help)");
        assert_eq!(err(&["/test"]), "unknown argument '/test' (see --help)");
    }

    #[test]
    fn c_style_numbers() {
        assert_eq!(parse_long("42"), Some(42));
        assert_eq!(parse_long(" 42"), Some(42));
        assert_eq!(parse_long("+42"), Some(42));
        assert_eq!(parse_long("-42"), Some(-42));
        assert_eq!(parse_long("42 "), None);
        assert_eq!(parse_long(""), None);
        assert_eq!(parse_long("-"), None);
        assert_eq!(parse_long("0x10"), None);
        assert_eq!(parse_long("99999999999999999999"), None);
        assert_eq!(parse(&["-c", " 64"]).unwrap().n_ctx, 64);
        assert_eq!(parse(&["--temp", "5e-1"]).unwrap().sampler.temperature, 0.5);
        assert_eq!(parse(&["--temp", ".25"]).unwrap().sampler.temperature, 0.25);
    }

    #[test]
    fn language_pairs() {
        let pair = |s: &str| parse_language_pair(s).map(|(a, b)| format!("{a}|{b}"));
        assert_eq!(pair("en:de").as_deref(), Some("en|de"));
        assert_eq!(pair("pt-BR:zh_TW").as_deref(), Some("pt-BR|zh_TW"));
        assert_eq!(pair("English:German").as_deref(), Some("English|German"));
        for bad in ["", "en", ":de", "en:", ":", "en:de:fr", "en::de"] {
            assert_eq!(pair(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn help_text() {
        let text = usage("nncpu");
        assert!(text.starts_with(&format!("nncpu {} - run Gemma", nncpu::version())));
        assert!(text.contains("usage: nncpu [options]            pick a model"));
        assert!(text.contains("$NNCPU_THREADS"));
        assert!(text.contains("prefix for the task: none, query, document, similarity"));
        assert!(!text.contains("llmcpu") && !text.contains("--test"));
        assert!(text.ends_with("~/models.\n"));
    }
}
