//! The chat REPL, translation turns and one-shot prompts.

use crate::cli::{self, Options};
use crate::embed;
use crate::models::{self, file_name};
use crate::platform::{self, InterruptScope};
use crate::ui::{self, ProgressBar, format_bytes, ui_error, ui_info};
use nncpu::{
    ErrorKind, GenerateParams, GenerateResult, KvType, Model, ModelParams, Piece, Session, SessionParams, StopReason,
    WeightFormat,
};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Once;
use std::time::Instant;

/// Load `path` with a progress bar; Ctrl-C cancels. None after reporting the
/// failure.
pub fn load_model_ui(path: &Path, weights: WeightFormat) -> Option<Model> {
    note_slow_cpu();
    let mut bar = ProgressBar::new(format!("Loading {}", file_name(path)));
    let t0 = Instant::now();
    let loaded = {
        let _scope = InterruptScope::begin(); // Ctrl-C cancels the load
        Model::load_with_progress(path, ModelParams { weights }, |p| {
            bar.update(f64::from(p), "");
            !platform::interrupted()
        })
    };
    match loaded {
        Ok(model) => {
            let info = model.info();
            let bytes = if info.resident_bytes > 0 {
                info.resident_bytes
            } else {
                info.file_bytes
            };
            let secs = t0.elapsed().as_secs_f64();
            bar.finish(&format!("done in {secs:.1} s ({} in memory)", format_bytes(bytes)));
            Some(model)
        }
        Err(e) if e.kind() == ErrorKind::Cancelled => {
            bar.finish("cancelled");
            None
        }
        Err(e) => {
            bar.finish("failed");
            ui_error!("{e}");
            None
        }
    }
}

/// The library's fast kernels need AVX2 and FMA; without them it runs
/// portable ones. Said once, before the first model loads.
fn note_slow_cpu() {
    static ONCE: Once = Once::new();
    if !nncpu::cpu_supported() {
        ONCE.call_once(|| {
            ui_info!(
                "nncpu: this CPU lacks AVX2 and FMA (Intel Haswell / AMD Excavator or newer have them), so nncpu \
                 runs its portable kernels, which are many times slower"
            )
        });
    }
}

/// Where the reply is going, and what is on screen so far.
#[derive(Default)]
struct Stream {
    /// A person is watching: show every token at once.
    flush_each: bool,
    /// The reply so far does not end in a newline.
    line_open: bool,
    /// Reasoning is on screen, and the answer has not started.
    thinking: bool,
    /// ...and the reasoning does not end in a newline.
    thought_open: bool,
}

impl Stream {
    fn on_piece(&mut self, piece: Piece) -> bool {
        match piece {
            // The model's reasoning goes to stderr, dimmed, so that stdout
            // carries only the answer.
            Piece::Thinking(text) => {
                self.thinking = true;
                let styled = format!("{}{text}{}", ui::dim(), ui::reset());
                io::stderr().write_all(styled.as_bytes()).ok();
                self.thought_open = !text.is_empty() && !text.ends_with('\n');
            }
            Piece::Text(text) => {
                self.end_thinking(true);
                let mut stdout = io::stdout().lock();
                stdout.write_all(text.as_bytes()).ok();
                if self.flush_each {
                    stdout.flush().ok();
                }
                self.line_open = !text.is_empty() && !text.ends_with('\n');
            }
        }
        true
    }

    /// End the reasoning's last line, and leave a blank line before an answer.
    fn end_thinking(&mut self, answer_follows: bool) {
        if !self.thinking {
            return;
        }
        let end = match (self.thought_open, answer_follows) {
            (true, true) => "\n\n",
            (true, false) | (false, true) => "\n",
            (false, false) => "",
        };
        io::stderr().write_all(end.as_bytes()).ok();
        self.thinking = false;
        self.thought_open = false;
    }
}

/// What a message becomes: a chat turn, or with both languages set, a request
/// to translate it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct TurnMode {
    from: String,
    to: String,
}

impl TurnMode {
    fn translating(&self) -> bool {
        !self.from.is_empty()
    }
}

/// One chat (or translation) turn with Ctrl-C wired to stop it. When the
/// conversation has filled the context, it starts a fresh one and tries
/// again.
fn chat_turn(
    session: &mut Session,
    message: &str,
    mode: &TurnMode,
    params: &GenerateParams,
    out: &mut Stream,
) -> nncpu::Result<GenerateResult> {
    let cancel = session.cancel_handle();
    let _scope = InterruptScope::with_hook(move || cancel.cancel());
    let mut once = |s: &mut Session| {
        if mode.translating() {
            s.translate(message, &mode.from, &mode.to, params, |piece| out.on_piece(piece))
        } else {
            s.chat(message, params, |piece| out.on_piece(piece))
        }
    };
    let mut result = once(session);
    if matches!(&result, Err(e) if e.kind() == ErrorKind::ContextFull) && session.n_past() > 0 {
        ui_info!(
            "{}(the conversation filled the {}-token context, so it starts afresh){}",
            ui::dim(),
            session.n_ctx(),
            ui::reset()
        );
        session.reset();
        result = once(session);
    }
    result
}

/// The line after a reply: tokens, speed, and why it stopped if that was not
/// the model's own choice.
fn report_turn(result: &nncpu::Result<GenerateResult>, out: &mut Stream, o: &Options, show_stats: bool) {
    out.end_thinking(false);
    {
        let mut stdout = io::stdout().lock();
        if out.line_open {
            stdout.write_all(b"\n").ok();
        }
        stdout.flush().ok();
    }
    out.line_open = false;
    let r = match result {
        Ok(r) => r,
        Err(e) if e.kind() == ErrorKind::Cancelled => {
            ui_info!("{}[stopped]{}", ui::dim(), ui::reset());
            return;
        }
        Err(e) => {
            ui_error!("{e}");
            return;
        }
    };

    let note = match r.stop {
        StopReason::MaxTokens => format!(", cut at -n {}", o.n_predict),
        StopReason::Cancelled => ", stopped".into(),
        StopReason::ContextFull => ", context full: the next message starts afresh".into(),
        _ => String::new(),
    };
    if show_stats || !note.is_empty() {
        let decode = r.decode.as_secs_f64();
        let tps = if decode > 0.0 {
            r.n_generated as f64 / decode
        } else {
            0.0
        };
        let thought = if r.n_thinking > 0 {
            format!(" ({} of them thinking)", r.n_thinking)
        } else {
            String::new()
        };
        if o.verbose {
            let drafts = if r.n_drafted > 0 {
                format!("; {} of {} drafted tokens kept", r.n_draft_accepted, r.n_drafted)
            } else {
                String::new()
            };
            ui_info!(
                "{}[{} tokens{thought}, {tps:.1} tok/s; prompt {} tokens in {:.2} s{drafts}{note}]{}",
                ui::dim(),
                r.n_generated,
                r.n_prompt,
                r.prefill.as_secs_f64(),
                ui::reset()
            );
        } else {
            ui_info!(
                "{}[{} tokens{thought}, {tps:.1} tok/s{note}]{}",
                ui::dim(),
                r.n_generated,
                ui::reset()
            );
        }
    }
    if o.perf {
        eprint!("{}", nncpu::perf_report("reply", r.n_prompt + r.n_generated));
        nncpu::perf_reset();
    }
}

fn print_help() {
    ui_info!(
        "Commands:\n  \
         /reset            forget the conversation and start a new one\n  \
         /translate EN:DE  translate each message from one language to another\n  \
         /translate off    chat again\n  \
         /exit             quit (so do Ctrl-D, or Ctrl-Z then Enter on Windows)\n  \
         /help             this list\n\
         Ctrl-C stops a reply that is being written."
    );
}

/// What a line typed at the prompt asks for.
#[derive(Debug, PartialEq, Eq)]
enum Input<'a> {
    Empty,
    Exit,
    Help,
    Reset,
    /// /translate off
    StopTranslating,
    /// /translate FROM:TO
    Translate(String, String),
    /// /translate with something else after it.
    TranslateUsage,
    UnknownCommand(&'a str),
    Message(&'a str),
}

fn parse_input(line: &str) -> Input<'_> {
    let msg = trim(line);
    match msg {
        "" => return Input::Empty,
        "/exit" | "/quit" | "/bye" => return Input::Exit,
        "/help" | "/?" => return Input::Help,
        "/reset" | "/clear" => return Input::Reset,
        _ => {}
    }
    if let Some(rest) = msg.strip_prefix("/translate")
        && (rest.is_empty() || rest.starts_with(' '))
    {
        let arg = trim(rest);
        if arg == "off" {
            return Input::StopTranslating;
        }
        return match cli::parse_language_pair(arg) {
            Some((from, to)) => Input::Translate(from, to),
            None => Input::TranslateUsage,
        };
    }
    if msg.starts_with('/') && !msg.contains(' ') {
        return Input::UnknownCommand(msg);
    }
    Input::Message(msg)
}

/// Strip surrounding blanks.
fn trim(s: &str) -> &str {
    s.trim_matches([' ', '\t'])
}

/// Pick, download or open the model, then chat: interactively, or answer
/// `o.prompt` once.
pub fn run_chat(o: &Options) -> u8 {
    let Some(path) = models::resolve_model(&o.model, o.yes) else {
        return 1;
    };
    let Some(model) = load_model_ui(&path, o.weights) else {
        return 1;
    };
    let info = model.info();
    if info.n_embd_out > 0 {
        return embed::run_embed(o, model);
    }

    // Gemma 4 drafts with its MTP "assistant" model when one is around.
    let mut draft_path = PathBuf::from(&o.draft);
    if !o.no_draft && o.draft.is_empty() && info.architecture == "gemma4" {
        draft_path = models::find_draft_model(&path).unwrap_or_default();
    }
    let mut draft = None;
    if !o.no_draft && !draft_path.as_os_str().is_empty() {
        draft = load_model_ui(&draft_path, WeightFormat::Q8_0);
        if draft.is_none() && !o.draft.is_empty() {
            return 1;
        }
    }

    let mut params = SessionParams {
        n_ctx: o.n_ctx,
        seed: o.seed,
        system_prompt: o.system_prompt.clone(),
        draft_model: draft.clone(),
        n_draft: o.n_draft,
        kv_type: if o.kv_f32 { KvType::F32 } else { KvType::Bf16 },
        thinking: o.think,
    };
    let session = match Session::new(&model, &params) {
        // A drafter that was only found, not asked for, is not worth failing over.
        Err(e) if draft.is_some() && o.draft.is_empty() => {
            ui_info!(
                "{}(not drafting with {}: {e}){}",
                ui::dim(),
                file_name(&draft_path),
                ui::reset()
            );
            draft = None;
            params.draft_model = None;
            Session::new(&model, &params)
        }
        created => created,
    };
    let mut session = match session {
        Ok(session) => session,
        Err(e) => {
            ui_error!("{e}");
            return 1;
        }
    };

    let mut out = Stream {
        flush_each: platform::stdout_is_terminal(),
        ..Stream::default()
    };
    let gen_params = GenerateParams {
        max_tokens: o.n_predict,
        sampler: o.sampler,
    };
    if o.think && !info.has_thinking {
        ui_info!(
            "{}({} has no thinking mode, so --think changes nothing){}",
            ui::dim(),
            info.name,
            ui::reset()
        );
    }
    if o.perf {
        nncpu::perf_enable(true);
        nncpu::perf_reset();
    }

    let mut mode = TurnMode {
        from: o.translate_from.clone(),
        to: o.translate_to.clone(),
    };
    if !o.prompt.is_empty() {
        let result = chat_turn(&mut session, &o.prompt, &mode, &gen_params, &mut out);
        report_turn(&result, &mut out, o, o.verbose);
        return if result.is_ok() { 0 } else { 1 };
    }

    let drafting = if draft.is_some() {
        format!(", drafting with {}", file_name(&draft_path))
    } else {
        String::new()
    };
    ui_info!(
        "{}{}{} ({}, {} layers, {} weights) - {} threads, {}-token context{}{drafting}",
        ui::bold(),
        info.name,
        ui::reset(),
        info.architecture,
        info.n_layer,
        info.weights.name(),
        nncpu::num_threads(),
        session.n_ctx(),
        if o.think && info.has_thinking { ", thinking" } else { "" }
    );
    if mode.translating() {
        ui_info!(
            "Each message is translated from {} to {}; /translate off to chat instead.",
            mode.from,
            mode.to
        );
    } else if info.translator {
        ui_info!(
            "This is a translation model: /translate en:de (or start with --translate en:de) translates each \
             message from English to German, and so on."
        );
    } else if !info.has_chat_template {
        ui_info!("nncpu does not know this model's chat format, so your text goes to it as it is.");
    }
    if draft.is_none()
        && !o.no_draft
        && let Some(e) = models::find_catalog_drafter(&path)
    {
        ui_info!(
            "{}Tip: nncpu --download {} fetches a drafter that makes this model reply faster.{}",
            ui::dim(),
            e.name,
            ui::reset()
        );
    }
    if info.has_thinking && !o.think && !mode.translating() {
        ui_info!(
            "{}Tip: --think lets this model reason before it answers: slower, better at hard problems.{}",
            ui::dim(),
            ui::reset()
        );
    }
    ui_info!(
        "{}Type a message and press Enter. /help lists commands; Ctrl-C stops a reply.{}",
        ui::dim(),
        ui::reset()
    );

    let tty = platform::stdin_is_terminal();
    loop {
        io::stdout().flush().ok();
        if tty {
            eprint!("\n{}>{} ", ui::bold(), ui::reset());
        }
        let Some(line) = platform::read_line() else {
            if tty {
                eprintln!();
            }
            break;
        };
        match parse_input(&line) {
            Input::Empty => {}
            Input::Exit => break,
            Input::Help => print_help(),
            Input::Reset => {
                session.reset();
                ui_info!("{}(new conversation){}", ui::dim(), ui::reset());
            }
            Input::StopTranslating => {
                mode = TurnMode::default();
                ui_info!("{}(chatting){}", ui::dim(), ui::reset());
            }
            Input::Translate(from, to) => {
                ui_info!("{}(translating from {from} to {to}){}", ui::dim(), ui::reset());
                mode = TurnMode { from, to };
            }
            Input::TranslateUsage => {
                ui_info!("Use /translate FROM:TO with language codes, e.g. /translate en:de, or /translate off.")
            }
            Input::UnknownCommand(command) => ui_info!("Unknown command {command}; /help lists the commands."),
            Input::Message(message) => {
                let result = chat_turn(&mut session, message, &mode, &gen_params, &mut out);
                report_turn(&result, &mut out, o, true);
            }
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_lines() {
        assert_eq!(parse_input(""), Input::Empty);
        assert_eq!(parse_input("   "), Input::Empty);
        assert_eq!(parse_input("/exit"), Input::Exit);
        assert_eq!(parse_input(" /quit\t"), Input::Exit);
        assert_eq!(parse_input("/bye"), Input::Exit);
        assert_eq!(parse_input("/help"), Input::Help);
        assert_eq!(parse_input("/?"), Input::Help);
        assert_eq!(parse_input("/reset"), Input::Reset);
        assert_eq!(parse_input("/clear"), Input::Reset);
        assert_eq!(parse_input("/hlep"), Input::UnknownCommand("/hlep"));
        assert_eq!(
            parse_input("/usr/bin is a folder"),
            Input::Message("/usr/bin is a folder")
        );
        assert_eq!(
            parse_input("  Why is the sky blue?  "),
            Input::Message("Why is the sky blue?")
        );
        assert_eq!(parse_input("/exit now"), Input::Message("/exit now"));
    }

    #[test]
    fn translate_command() {
        assert_eq!(
            parse_input("/translate en:de"),
            Input::Translate("en".into(), "de".into())
        );
        assert_eq!(
            parse_input("/translate   pt-BR:ja  "),
            Input::Translate("pt-BR".into(), "ja".into())
        );
        assert_eq!(parse_input("/translate off"), Input::StopTranslating);
        assert_eq!(parse_input("/translate  off "), Input::StopTranslating);
        assert_eq!(parse_input("/translate"), Input::TranslateUsage);
        assert_eq!(parse_input("/translate en"), Input::TranslateUsage);
        assert_eq!(parse_input("/translate en:de:fr"), Input::TranslateUsage);
        assert_eq!(parse_input("/translate :de"), Input::TranslateUsage);
        // The languages go to the library as typed; it trims them.
        assert_eq!(
            parse_input("/translate en: de"),
            Input::Translate("en".into(), " de".into())
        );
        assert_eq!(parse_input("/translateen:de"), Input::UnknownCommand("/translateen:de"));
        assert_eq!(
            parse_input("/translate\ten:de"),
            Input::UnknownCommand("/translate\ten:de")
        );
    }

    #[test]
    fn turn_modes() {
        assert!(!TurnMode::default().translating());
        assert!(
            TurnMode {
                from: "en".into(),
                to: "de".into()
            }
            .translating()
        );
    }
}
