//! The library's log messages on stderr: warnings and errors always, as
//! "nncpu: ..." and "nncpu: error: ..."; with -v, its info and debug
//! messages too, dimmed.

use crate::ui::{self, ui_info};
use log::{Level, LevelFilter, Log, Metadata, Record};
use std::fmt;

struct Logger;

static LOGGER: Logger = Logger;

impl Log for Logger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= log::max_level()
    }

    fn log(&self, record: &Record) {
        if self.enabled(record.metadata()) {
            ui_info!("{}", shown(record.level(), record.args()));
        }
    }

    fn flush(&self) {}
}

/// How a message from the library is shown.
fn shown(level: Level, message: &fmt::Arguments) -> String {
    match level {
        Level::Error => format!("nncpu: error: {message}"),
        Level::Warn => format!("nncpu: {message}"),
        _ => format!("{}{message}{}", ui::dim(), ui::reset()),
    }
}

/// Install the logger; `verbose` (-v) shows what the library is doing.
pub fn init(verbose: bool) {
    if log::set_logger(&LOGGER).is_ok() {
        log::set_max_level(if verbose { LevelFilter::Debug } else { LevelFilter::Warn });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages() {
        // Colour is off: the console was never set up.
        assert_eq!(
            shown(Level::Error, &format_args!("cannot open 'x'")),
            "nncpu: error: cannot open 'x'"
        );
        assert_eq!(shown(Level::Warn, &format_args!("odd file")), "nncpu: odd file");
        assert_eq!(
            shown(Level::Info, &format_args!("context: 4096 tokens")),
            "context: 4096 tokens"
        );
        assert_eq!(shown(Level::Debug, &format_args!("{} tokens", 3)), "3 tokens");
    }
}
