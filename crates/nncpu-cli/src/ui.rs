//! Terminal output: messages, colour, progress bars, questions and number
//! formatting. Status goes to stderr so that stdout carries only what the
//! model writes (and the diagnostic modes' reports).

use crate::platform;
use std::fmt;
use std::io::{self, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// A progress bar has drawn a line that does not end in a newline yet.
static BAR_OPEN: AtomicBool = AtomicBool::new(false);

/// ANSI styles, or "" when colour is off.
pub fn dim() -> &'static str {
    style("\x1b[2m")
}

pub fn bold() -> &'static str {
    style("\x1b[1m")
}

pub fn red() -> &'static str {
    style("\x1b[31m")
}

pub fn reset() -> &'static str {
    style("\x1b[0m")
}

fn style(sequence: &'static str) -> &'static str {
    if platform::color() { sequence } else { "" }
}

/// One line on stderr: `prefix`, the message, and an end to any style.
fn line(prefix: &str, message: fmt::Arguments) {
    io::stdout().flush().ok(); // keep model text and messages in order
    // Don't write over a half-drawn bar.
    let newline = if BAR_OPEN.swap(false, Ordering::Relaxed) {
        "\n"
    } else {
        ""
    };
    let text = format!("{newline}{prefix}{message}{}\n", reset());
    io::stderr().write_all(text.as_bytes()).ok();
}

pub fn info(message: fmt::Arguments) {
    line("", message);
}

pub fn error(message: fmt::Arguments) {
    line(&format!("{}nncpu: error: ", red()), message);
}

/// A line on stderr, formatted like `println!`.
macro_rules! ui_info {
    ($($arg:tt)*) => {
        $crate::ui::info(format_args!($($arg)*))
    };
}

/// "nncpu: error: ..." on stderr, in red.
macro_rules! ui_error {
    ($($arg:tt)*) => {
        $crate::ui::error(format_args!($($arg)*))
    };
}

/// `print!` that shrugs off a closed stdout (`nncpu --gguf-info x | head`).
macro_rules! out {
    ($($arg:tt)*) => {
        $crate::ui::write_out(format_args!($($arg)*))
    };
}

/// `println!` that shrugs off a closed stdout.
macro_rules! outln {
    () => {
        $crate::ui::write_out(format_args!("\n"))
    };
    ($($arg:tt)*) => {
        $crate::ui::write_out(format_args!("{}\n", format_args!($($arg)*)))
    };
}

pub(crate) use {out, outln, ui_error, ui_info};

pub fn write_out(text: fmt::Arguments) {
    io::stdout().write_fmt(text).ok();
}

/// "1.0 GB", "512 MB", "3 KB", "12 B".
pub fn format_bytes(bytes: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;
    let b = bytes as f64;
    if b >= GB {
        format!("{:.1} GB", b / GB)
    } else if b >= MB {
        format!("{:.0} MB", b / MB)
    } else if b >= KB {
        format!("{:.0} KB", b / KB)
    } else {
        format!("{bytes} B")
    }
}

/// "0:42", "1:05:09".
pub fn format_duration(seconds: f64) -> String {
    let s = (seconds + 0.5) as u64;
    if s >= 3600 {
        format!("{}:{:02}:{:02}", s / 3600, (s / 60) % 60, s % 60)
    } else {
        format!("{}:{:02}", s / 60, s % 60)
    }
}

/// `path` with the home directory written as "~".
pub fn tidy_path(path: &Path) -> String {
    let path = path.display().to_string();
    match platform::home_dir() {
        Some(home) => tilde(&path, &home.display().to_string()),
        None => path,
    }
}

fn tilde(path: &str, home: &str) -> String {
    if !home.is_empty()
        && path.len() > home.len()
        && path.starts_with(home)
        && matches!(path.as_bytes()[home.len()], b'/' | b'\\')
    {
        format!("~{}", &path[home.len()..])
    } else {
        path.to_string()
    }
}

/// Ask a yes/no question on stderr and read the answer; `fallback` is the
/// answer for an empty line, and for the end of input.
pub fn confirm(question: &str, fallback: bool) -> bool {
    io::stdout().flush().ok();
    eprint!("{question} {} ", if fallback { "[Y/n]" } else { "[y/N]" });
    platform::read_line().map_or(fallback, |answer| yes_or_no(&answer, fallback))
}

/// The answer a line gives: its first non-blank character, y or n, else
/// `fallback`.
fn yes_or_no(answer: &str, fallback: bool) -> bool {
    match answer.trim_start_matches([' ', '\t']).chars().next() {
        Some('y' | 'Y') => true,
        Some('n' | 'N') => false,
        _ => fallback,
    }
}

/// A one-line progress bar on stderr, redrawn in place. Silent when stderr
/// is not a terminal.
pub struct ProgressBar {
    label: String,
    live: bool,
    drawn: bool,
    last: f64,
    last_at: Instant,
}

impl ProgressBar {
    pub fn new(label: impl Into<String>) -> ProgressBar {
        ProgressBar {
            label: label.into(),
            live: platform::stderr_is_terminal(),
            drawn: false,
            last: -1.0,
            last_at: Instant::now(),
        }
    }

    /// `fraction` in [0, 1]; `detail` follows the bar ("456 MB / 1013 MB ...").
    pub fn update(&mut self, fraction: f64, detail: &str) {
        if !self.live {
            return;
        }
        let f = fraction.clamp(0.0, 1.0);
        let now = Instant::now();
        // At most ten redraws a second, plus the moment it completes.
        if self.drawn && f < 1.0 && (now - self.last_at < Duration::from_millis(100) || f - self.last < 0.001) {
            return;
        }
        self.last = f;
        self.last_at = now;
        const WIDTH: usize = 24;
        let filled = ((f * WIDTH as f64 + 0.5) as usize).min(WIDTH);
        let text = format!(
            "\r{} [{}{}] {:>3}%{}{}{}",
            self.label,
            "#".repeat(filled),
            "-".repeat(WIDTH - filled),
            (f * 100.0 + 0.5) as u32,
            if detail.is_empty() { "" } else { "  " },
            detail,
            if platform::color() { "\x1b[K" } else { "   " }
        );
        io::stderr().write_all(text.as_bytes()).ok();
        self.drawn = true;
        BAR_OPEN.store(true, Ordering::Relaxed);
    }

    /// Finish the line, replacing the bar with `summary` when one is given.
    pub fn finish(&mut self, summary: &str) {
        if self.live && self.drawn && BAR_OPEN.load(Ordering::Relaxed) {
            let clear = if platform::color() {
                "\x1b[K".to_string()
            } else {
                " ".repeat(50)
            };
            eprint!("\r{} {summary}{clear}\n", self.label);
        } else if !summary.is_empty() {
            eprintln!("{} {summary}", self.label);
        }
        if self.live {
            self.drawn = false;
            BAR_OPEN.store(false, Ordering::Relaxed);
        }
    }
}

/// `v` as C's printf("%.*g", precision, v) writes it: the shorter of fixed and
/// exponent notation at `precision` significant digits, trailing zeros removed.
pub fn fmt_g(v: f64, precision: usize) -> String {
    let p = precision.max(1);
    if v.is_nan() {
        return "nan".into();
    }
    if v.is_infinite() {
        return if v < 0.0 { "-inf" } else { "inf" }.into();
    }
    if v == 0.0 {
        return if v.is_sign_negative() { "-0" } else { "0" }.into();
    }
    // The exponent is the one the value has once rounded to p digits.
    let sci = format!("{:.*e}", p - 1, v);
    let (mantissa, exp) = sci.split_once('e').expect("exponent notation has an 'e'");
    let exp: i32 = exp.parse().expect("the exponent is an integer");
    if exp < -4 || exp >= p as i32 {
        format!(
            "{}e{}{:02}",
            trim_fraction(mantissa),
            if exp < 0 { '-' } else { '+' },
            exp.unsigned_abs()
        )
    } else {
        trim_fraction(&format!("{:.*}", (p as i32 - 1 - exp) as usize, v)).to_string()
    }
}

/// printf("%+.*g"): `fmt_g` with a sign on positive values too.
pub fn fmt_g_signed(v: f64, precision: usize) -> String {
    let s = fmt_g(v, precision);
    if s.starts_with('-') { s } else { format!("+{s}") }
}

/// "1.500" -> "1.5", "2.000" -> "2"; whole numbers are left alone.
fn trim_fraction(s: &str) -> &str {
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.')
    } else {
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_and_durations() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(1023), "1023 B");
        assert_eq!(format_bytes(1024), "1 KB");
        assert_eq!(format_bytes(536_576), "524 KB");
        assert_eq!(format_bytes(97_817_696), "93 MB");
        assert_eq!(format_bytes(1_011_738_656), "965 MB");
        assert_eq!(format_bytes(16_796_017_248), "15.6 GB");
        assert_eq!(format_duration(0.0), "0:00");
        assert_eq!(format_duration(42.4), "0:42");
        assert_eq!(format_duration(59.6), "1:00");
        assert_eq!(format_duration(3909.0), "1:05:09");
    }

    #[test]
    fn home_becomes_tilde() {
        assert_eq!(tilde(r"C:\Users\me\models", r"C:\Users\me"), r"~\models");
        assert_eq!(tilde("/home/me/models/x.gguf", "/home/me"), "~/models/x.gguf");
        assert_eq!(tilde("/home/meg/models", "/home/me"), "/home/meg/models");
        assert_eq!(tilde("/home/me", "/home/me"), "/home/me");
        assert_eq!(tilde("models", ""), "models");
    }

    #[test]
    fn yes_no_answers() {
        assert!(yes_or_no("y", false));
        assert!(yes_or_no("  Yes please", false));
        assert!(!yes_or_no("n", true));
        assert!(!yes_or_no("\tNo", true));
        assert!(yes_or_no("", true));
        assert!(!yes_or_no("", false));
        assert!(yes_or_no("maybe", true));
    }

    #[test]
    fn printf_g() {
        // What C's printf prints for each.
        let cases: &[(f64, usize, &str)] = &[
            (1e-6, 6, "1e-06"),
            (1e6, 6, "1e+06"),
            (1e7, 6, "1e+07"),
            (10000.0, 6, "10000"),
            (100000.0, 6, "100000"),
            (123456789.0, 6, "1.23457e+08"),
            (0.0001, 6, "0.0001"),
            (0.00001234, 6, "1.234e-05"),
            (1.234_567_89, 6, "1.23457"),
            (9.9999999, 6, "10"),
            (0.5, 6, "0.5"),
            (-2.5, 6, "-2.5"),
            (0.0, 6, "0"),
            (0.885_3, 4, "0.8853"),
            (2.0, 4, "2"),
            (0.000_614_5, 4, "0.0006145"),
            (-0.001_369, 4, "-0.001369"),
            (1.0, 0, "1"),
            (0.123_456_78, 7, "0.1234568"),
            (-1.5e-10, 7, "-1.5e-10"),
            (f64::INFINITY, 6, "inf"),
            (f64::NEG_INFINITY, 4, "-inf"),
        ];
        for &(v, p, want) in cases {
            assert_eq!(fmt_g(v, p), want, "fmt_g({v}, {p})");
        }
        assert_eq!(fmt_g_signed(2.0, 4), "+2");
        assert_eq!(fmt_g_signed(-0.07358, 4), "-0.07358");
        assert_eq!(fmt_g_signed(0.0, 4), "+0");
    }
}
