//! Text-embedding models (EmbeddingGemma): vectors for scripts, and an
//! interactive similarity explorer.

use crate::cli::Options;
use crate::platform;
use crate::ui::{self, fmt_g, out, outln, ui_error, ui_info};
use nncpu::Model;
use std::io::{self, Write};
use std::time::Instant;

/// EmbeddingGemma's prompts, from its model card.
const TASKS: &[(&str, &str)] = &[
    ("none", ""),
    ("query", "task: search result | query: "),
    ("document", "title: none | text: "),
    ("similarity", "task: sentence similarity | query: "),
    ("classification", "task: classification | query: "),
    ("clustering", "task: clustering | query: "),
    ("qa", "task: question answering | query: "),
    ("fact", "task: fact checking | query: "),
    ("code", "task: code retrieval | query: "),
];

/// The prefix EmbeddingGemma expects for a task ("query", "document",
/// "similarity", ...), "" for "none", or None for an unknown name.
pub fn task_prefix(name: &str) -> Option<&'static str> {
    TASKS.iter().find(|(task, _)| *task == name).map(|&(_, prefix)| prefix)
}

/// The task names, comma separated, for messages.
pub fn task_names() -> String {
    TASKS.iter().map(|&(task, _)| task).collect::<Vec<_>>().join(", ")
}

/// Run an embedding model that run_chat() has loaded. -p TEXT or piped lines
/// print one JSON array per line; typed lines are compared with the lines
/// before them.
pub fn run_embed(o: &Options, model: Model) -> u8 {
    let info = model.info();
    let n_out = if o.dim > 0 { o.dim } else { info.n_embd_out };
    if n_out > info.n_embd_out {
        ui_error!(
            "{} makes vectors of {} values; --dim {} asks for more",
            info.name,
            info.n_embd_out,
            o.dim
        );
        return 1;
    }

    // Scripts get one vector per line of input, in order; a person typing
    // gets similarities, for which the "similarity" prompt suits best.
    let interactive = o.prompt.is_empty() && platform::stdin_is_terminal();
    let task = if !o.task.is_empty() {
        o.task.as_str()
    } else if interactive {
        "similarity"
    } else {
        "none"
    };
    let prefix = task_prefix(task).unwrap_or_default();
    let mut vec = vec![0.0; n_out];
    let embed = |text: &str, vec: &mut [f32]| match model.embed(&format!("{prefix}{text}"), vec) {
        Ok(()) => true,
        Err(e) => {
            ui_error!("{e}");
            false
        }
    };

    if !interactive {
        let mut rc = 0;
        if !o.prompt.is_empty() {
            if embed(&o.prompt, &mut vec) {
                out!("{}", json_line(&vec));
            } else {
                rc = 1;
            }
        } else {
            while rc == 0
                && let Some(line) = platform::read_line()
            {
                if embed(&line, &mut vec) {
                    out!("{}", json_line(&vec));
                } else {
                    rc = 1;
                }
            }
        }
        io::stdout().flush().ok();
        return rc;
    }

    ui_info!(
        "{}{}{} ({}, {} layers) - a text-embedding model: {n_out} values per text, task \"{task}\"",
        ui::bold(),
        info.name,
        ui::reset(),
        info.architecture,
        info.n_layer
    );
    ui_info!(
        "{}Type a line to see how close in meaning it is to the lines before it. /help lists commands.{}",
        ui::dim(),
        ui::reset()
    );

    let mut seen: Vec<(String, Vec<f32>)> = Vec::new();
    loop {
        io::stdout().flush().ok();
        eprint!("\n{}>{} ", ui::bold(), ui::reset());
        let Some(line) = platform::read_line() else {
            eprintln!();
            break;
        };
        match parse_input(&line) {
            Input::Empty => {}
            Input::Exit => break,
            Input::Help => print_help(),
            Input::Reset => {
                seen.clear();
                ui_info!("{}(forgotten){}", ui::dim(), ui::reset());
            }
            Input::List => {
                for (i, (text, _)) in seen.iter().enumerate() {
                    outln!("  {}. {text}", i + 1);
                }
                if seen.is_empty() {
                    ui_info!("{}(no lines yet){}", ui::dim(), ui::reset());
                }
            }
            Input::Unknown(command) => ui_info!("Unknown command {command}; /help lists the commands."),
            Input::Text(text) => {
                let t0 = Instant::now();
                if !embed(text, &mut vec) {
                    continue;
                }
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                let ranked = rank(&vec, &seen);
                let shown = ranked.len().min(5);
                for &(similarity, i) in &ranked[..shown] {
                    out!("{}", similarity_line(similarity, &seen[i].0));
                }
                if ranked.len() > shown {
                    ui_info!("{}  ({} more){}", ui::dim(), ranked.len() - shown, ui::reset());
                }
                let hint = if seen.is_empty() {
                    "; type another line to compare it with this one"
                } else {
                    ""
                };
                ui_info!("{}[{n_out} values in {ms:.0} ms{hint}]{}", ui::dim(), ui::reset());
                if o.perf {
                    eprint!("{}", nncpu::perf_report("embedding", 0));
                    nncpu::perf_reset();
                }
                seen.push((text.to_string(), vec.clone()));
            }
        }
    }
    0
}

/// What a line typed at the prompt asks for.
#[derive(Debug, PartialEq, Eq)]
enum Input<'a> {
    Empty,
    Exit,
    Help,
    Reset,
    List,
    Unknown(&'a str),
    /// A line to embed and compare.
    Text(&'a str),
}

fn parse_input(line: &str) -> Input<'_> {
    let msg = line.trim_matches([' ', '\t']);
    match msg {
        "" => Input::Empty,
        "/exit" | "/quit" | "/bye" => Input::Exit,
        "/help" | "/?" => Input::Help,
        "/reset" | "/clear" => Input::Reset,
        "/list" => Input::List,
        _ if msg.starts_with('/') && !msg.contains(' ') => Input::Unknown(msg),
        _ => Input::Text(msg),
    }
}

fn print_help() {
    ui_info!(
        "Type a line to see how close in meaning it is to each line before it\n\
         (1 = the same meaning, around 0 = unrelated).\n\
         Commands:\n  \
         /list     the lines so far\n  \
         /reset    forget them\n  \
         /exit     quit (so do Ctrl-D, or Ctrl-Z then Enter on Windows)\n  \
         /help     this list"
    );
}

/// One vector as a line of JSON.
fn json_line(v: &[f32]) -> String {
    let values: Vec<String> = v.iter().map(|&x| fmt_g(f64::from(x), 7)).collect();
    format!("[{}]\n", values.join(","))
}

/// The vectors have unit length, so this is their cosine similarity.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(&x, &y)| f64::from(x) * f64::from(y)).sum::<f64>() as f32
}

/// The lines seen so far as (similarity to `v`, index), most similar first.
fn rank(v: &[f32], seen: &[(String, Vec<f32>)]) -> Vec<(f32, usize)> {
    let mut ranked: Vec<(f32, usize)> = seen.iter().enumerate().map(|(i, (_, w))| (dot(v, w), i)).collect();
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
    ranked
}

/// "   0.83  ████████████████      the line", with a 20-block bar.
fn similarity_line(similarity: f32, text: &str) -> String {
    let blocks = (similarity.clamp(0.0, 1.0) * 20.0 + 0.5) as usize;
    format!(
        "  {similarity:5.2}  {}{}  {}\n",
        "\u{2588}".repeat(blocks),
        " ".repeat(20 - blocks),
        clip(text, 70)
    )
}

/// At most `max_bytes` of `s`, cut at a character boundary, with "..." when
/// something was cut.
fn clip(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s.to_string();
    }
    let mut n = max_bytes;
    while !s.is_char_boundary(n) {
        n -= 1;
    }
    format!("{}...", &s[..n])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tasks() {
        assert_eq!(task_prefix("none"), Some(""));
        assert_eq!(task_prefix("query"), Some("task: search result | query: "));
        assert_eq!(task_prefix("document"), Some("title: none | text: "));
        assert_eq!(task_prefix("Query"), None);
        assert_eq!(task_prefix(""), None);
        assert_eq!(
            task_names(),
            "none, query, document, similarity, classification, clustering, qa, fact, code"
        );
    }

    #[test]
    fn typed_lines() {
        assert_eq!(parse_input(""), Input::Empty);
        assert_eq!(parse_input(" \t "), Input::Empty);
        assert_eq!(parse_input("/exit"), Input::Exit);
        assert_eq!(parse_input("  /bye "), Input::Exit);
        assert_eq!(parse_input("/?"), Input::Help);
        assert_eq!(parse_input("/clear"), Input::Reset);
        assert_eq!(parse_input("/list"), Input::List);
        assert_eq!(parse_input("/lsit"), Input::Unknown("/lsit"));
        assert_eq!(parse_input("/etc/hosts is a file"), Input::Text("/etc/hosts is a file"));
        assert_eq!(parse_input("  The cat sat.\t"), Input::Text("The cat sat."));
    }

    #[test]
    fn json_lines() {
        assert_eq!(json_line(&[]), "[]\n");
        assert_eq!(
            json_line(&[0.5, -0.25, 1e-8, 0.123_456_78]),
            "[0.5,-0.25,1e-08,0.1234568]\n"
        );
    }

    #[test]
    fn similarity_display() {
        let line = similarity_line(0.83, "the cat");
        assert_eq!(
            line,
            format!("   0.83  {}{}  the cat\n", "\u{2588}".repeat(17), " ".repeat(3))
        );
        let line = similarity_line(-0.1, "x");
        assert_eq!(line, format!("  -0.10  {}  x\n", " ".repeat(20)));
        let long = "é".repeat(40); // 80 bytes
        assert_eq!(clip(&long, 70), format!("{}...", "é".repeat(35)));
        assert_eq!(clip(&format!("a{long}"), 70), format!("a{}...", "é".repeat(34)));
        assert_eq!(clip("short", 70), "short");
    }

    #[test]
    fn ranking() {
        let seen = vec![
            ("a".to_string(), vec![1.0, 0.0]),
            ("b".into(), vec![0.0, 1.0]),
            ("c".into(), vec![0.6, 0.8]),
        ];
        let ranked = rank(&[0.0, 1.0], &seen);
        assert_eq!(ranked.iter().map(|&(_, i)| i).collect::<Vec<_>>(), [1, 2, 0]);
        assert!((ranked[1].0 - 0.8).abs() < 1e-6);
        assert!(rank(&[1.0, 0.0], &[]).is_empty());
    }
}
