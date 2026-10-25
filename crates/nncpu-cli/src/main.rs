//! nncpu, the command-line app: chat with a GGUF model on the CPU, or
//! download, list and examine models.
//!
//! Sets up the console, parses the options and hands over to the chat
//! (chat.rs), the model downloader (models.rs) or the diagnostic modes
//! (devtools.rs). Everything that runs a model goes through the nncpu
//! library's public API.

mod catalog;
mod chat;
mod cli;
mod devtools;
mod download;
mod embed;
mod logging;
mod models;
mod platform;
mod ui;

use cli::Mode;
use std::process::ExitCode;

fn main() -> ExitCode {
    platform::init_console();
    platform::install_interrupt_handler();

    let args: Vec<String> = std::env::args_os().map(|a| a.to_string_lossy().into_owned()).collect();
    let options = match cli::parse_options(args.get(1..).unwrap_or_default()) {
        Ok(options) => options,
        Err(problem) => {
            eprintln!("nncpu: {problem}");
            return ExitCode::from(1);
        }
    };
    logging::init(options.verbose);
    if options.n_threads > 0 {
        nncpu::set_num_threads(options.n_threads);
    }
    if options.perf {
        nncpu::perf_enable(true);
    }

    let code = match options.mode {
        Mode::Help => {
            ui::out!("{}", cli::usage(args.first().map_or("nncpu", String::as_str)));
            0
        }
        Mode::Version => {
            ui::outln!("nncpu {}", nncpu::version());
            0
        }
        Mode::ListModels => models::run_list_models(),
        Mode::Download => models::run_download(options.args.first().map_or("", String::as_str), options.yes),
        Mode::Chat => chat::run_chat(&options),
        _ => devtools::run_devtool(&options),
    };
    ExitCode::from(code)
}

/// A fresh, empty folder for a test's files, inside the build's target
/// folder; the test removes it when done.
#[cfg(test)]
pub(crate) fn test_dir(name: &str) -> std::path::PathBuf {
    let exe = std::env::current_exe().expect("the test binary has a path");
    let base = exe.parent().expect("it is in a folder").join("nncpu-cli-tests");
    let dir = base.join(format!("{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("the target folder is writable");
    dir
}
