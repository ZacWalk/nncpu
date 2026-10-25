//! The app's view of the OS: the console, line input, the home directory
//! and Ctrl-C.

use std::io::{self, BufRead, IsTerminal};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};

static COLOR: AtomicBool = AtomicBool::new(false);

/// One-time console setup: ANSI sequences on a Windows console, and whether
/// to colour stderr, which carries every styled message: when it is a
/// terminal that takes ANSI sequences and NO_COLOR is not set.
pub fn init_console() {
    let ansi = enable_ansi();
    let no_color = std::env::var_os("NO_COLOR").is_some_and(|v| !v.is_empty());
    COLOR.store(ansi && !no_color && io::stderr().is_terminal(), Ordering::Relaxed);
}

/// Turn on ANSI sequences in the console; false when stderr is a console that
/// cannot take them (one older than Windows 10).
#[cfg(windows)]
fn enable_ansi() -> bool {
    use std::ffi::c_void;
    const STD_OUTPUT_HANDLE: u32 = -11i32 as u32;
    const STD_ERROR_HANDLE: u32 = -12i32 as u32;
    const ENABLE_VIRTUAL_TERMINAL_PROCESSING: u32 = 0x0004;
    unsafe extern "system" {
        fn GetStdHandle(which: u32) -> *mut c_void;
        fn GetConsoleMode(console: *mut c_void, mode: *mut u32) -> i32;
        fn SetConsoleMode(console: *mut c_void, mode: u32) -> i32;
    }
    // None when the stream is not a console (a file, a pipe, or a mintty
    // terminal, which takes ANSI sequences as it is).
    let enable = |which: u32| {
        // SAFETY: plain Win32 calls on this process's standard handles; a
        // handle that is not a console just makes GetConsoleMode fail.
        unsafe {
            let console = GetStdHandle(which);
            let mut mode = 0;
            if GetConsoleMode(console, &mut mode) == 0 {
                return None;
            }
            Some(SetConsoleMode(console, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING) != 0)
        }
    };
    enable(STD_OUTPUT_HANDLE);
    enable(STD_ERROR_HANDLE).unwrap_or(true)
}

/// Terminals take ANSI sequences unless TERM says otherwise.
#[cfg(not(windows))]
fn enable_ansi() -> bool {
    std::env::var_os("TERM").is_some_and(|term| term != "dumb")
}

/// Whether stderr gets colour (see `init_console`).
pub fn color() -> bool {
    COLOR.load(Ordering::Relaxed)
}

pub fn stdin_is_terminal() -> bool {
    io::stdin().is_terminal()
}

pub fn stdout_is_terminal() -> bool {
    io::stdout().is_terminal()
}

pub fn stderr_is_terminal() -> bool {
    io::stderr().is_terminal()
}

/// One line of input without its line ending; None at the end of input
/// (Ctrl-D, Ctrl-Z on Windows, a closed pipe). A console is read as UTF-16
/// by the standard library, so any text survives; bytes from a pipe that are
/// not UTF-8 become U+FFFD.
pub fn read_line() -> Option<String> {
    let mut line = Vec::new();
    match io::stdin().lock().read_until(b'\n', &mut line) {
        Ok(0) | Err(_) => return None,
        Ok(_) => {}
    }
    if line.last() == Some(&b'\n') {
        line.pop();
    }
    if line.last() == Some(&b'\r') {
        line.pop();
    }
    Some(String::from_utf8_lossy(&line).into_owned())
}

/// The user's home directory (USERPROFILE on Windows, HOME elsewhere).
pub fn home_dir() -> Option<PathBuf> {
    let var = if cfg!(windows) { "USERPROFILE" } else { "HOME" };
    std::env::var_os(var).filter(|home| !home.is_empty()).map(PathBuf::from)
}

// ---- Ctrl-C -----------------------------------------------------------------
//
// Ctrl-C normally ends the process. While an InterruptScope is alive it
// instead sets interrupted() and calls the scope's hook (on the Ctrl-C
// handler's thread). A second Ctrl-C in the same scope ends the process.

type Hook = Box<dyn Fn() + Send>;

static SCOPES: AtomicUsize = AtomicUsize::new(0);
static INTERRUPTS: AtomicUsize = AtomicUsize::new(0);
static HOOK: Mutex<Option<Hook>> = Mutex::new(None);

/// Route Ctrl-C (and Ctrl-Break) through the interrupt scopes. Call once,
/// early.
pub fn install_interrupt_handler() {
    // Should this fail, Ctrl-C keeps its default of ending the process, which
    // is still right outside the scopes.
    let _ = ctrlc::set_handler(|| {
        if !on_interrupt() {
            exit_interrupted();
        }
    });
}

/// True when a scope handled the interrupt; false means "end the process as
/// usual".
fn on_interrupt() -> bool {
    if SCOPES.load(Ordering::SeqCst) == 0 || INTERRUPTS.fetch_add(1, Ordering::SeqCst) > 0 {
        return false;
    }
    if let Some(hook) = hook().as_ref() {
        hook();
    }
    true
}

/// End the process the way an unhandled Ctrl-C does: with
/// STATUS_CONTROL_C_EXIT on Windows, and elsewhere with the status a shell
/// reports for SIGINT.
fn exit_interrupted() -> ! {
    std::process::exit(if cfg!(windows) { 0xC000_013A_u32 as i32 } else { 130 })
}

fn hook() -> MutexGuard<'static, Option<Hook>> {
    HOOK.lock().unwrap_or_else(PoisonError::into_inner)
}

/// While alive, Ctrl-C stops the work at hand instead of ending nncpu.
#[must_use = "Ctrl-C is handled only while the scope is alive"]
pub struct InterruptScope(());

impl InterruptScope {
    /// Ctrl-C sets [`interrupted`], which the work in progress polls.
    pub fn begin() -> InterruptScope {
        InterruptScope::start(None)
    }

    /// Ctrl-C also calls `hook`, from another thread: e.g. a session's
    /// cancel handle.
    pub fn with_hook(hook: impl Fn() + Send + 'static) -> InterruptScope {
        InterruptScope::start(Some(Box::new(hook)))
    }

    fn start(new_hook: Option<Hook>) -> InterruptScope {
        INTERRUPTS.store(0, Ordering::SeqCst);
        *hook() = new_hook;
        SCOPES.fetch_add(1, Ordering::SeqCst);
        InterruptScope(())
    }
}

impl Drop for InterruptScope {
    fn drop(&mut self) {
        SCOPES.fetch_sub(1, Ordering::SeqCst);
        *hook() = None;
    }
}

/// True once Ctrl-C was pressed during the current (or last) scope.
pub fn interrupted() -> bool {
    INTERRUPTS.load(Ordering::SeqCst) > 0
}
