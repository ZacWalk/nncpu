//! Errors returned by nncpu.

use std::fmt;

/// The kind of failure an [`Error`] reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ErrorKind {
    /// An argument was out of range or otherwise invalid.
    InvalidArgument,
    /// A file could not be opened, mapped or read.
    Io,
    /// Not a GGUF file, or a corrupt one.
    BadFormat,
    /// An architecture or tensor type nncpu cannot run.
    UnsupportedModel,
    /// A buffer could not be allocated.
    OutOfMemory,
    /// The session's context window has no room.
    ContextFull,
    /// Stopped by a progress callback or a [`CancelHandle`](crate::CancelHandle).
    Cancelled,
}

impl ErrorKind {
    /// A short, static description, e.g. "context full".
    pub fn as_str(self) -> &'static str {
        match self {
            ErrorKind::InvalidArgument => "invalid argument",
            ErrorKind::Io => "I/O error",
            ErrorKind::BadFormat => "bad file format",
            ErrorKind::UnsupportedModel => "unsupported model",
            ErrorKind::OutOfMemory => "out of memory",
            ErrorKind::ContextFull => "context full",
            ErrorKind::Cancelled => "cancelled",
        }
    }
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// An nncpu failure: its [`ErrorKind`] and a specific message, such as
/// "cannot open 'x.gguf': the system cannot find the file specified".
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    kind: ErrorKind,
    message: String,
}

impl Error {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Error {
            kind,
            message: message.into(),
        }
    }

    pub fn kind(&self) -> ErrorKind {
        self.kind
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.message.is_empty() {
            f.write_str(self.kind.as_str())
        } else {
            f.write_str(&self.message)
        }
    }
}

impl std::error::Error for Error {}

impl From<ErrorKind> for Error {
    fn from(kind: ErrorKind) -> Self {
        Error {
            kind,
            message: String::new(),
        }
    }
}

/// `Result<T, nncpu::Error>`.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// `err!(Kind, "format", args...)` builds an [`Error`].
macro_rules! err {
    ($kind:ident, $($arg:tt)*) => {
        $crate::error::Error::new($crate::error::ErrorKind::$kind, format!($($arg)*))
    };
}

/// `bail!(Kind, "format", args...)` returns an [`Error`] from the function.
macro_rules! bail {
    ($kind:ident, $($arg:tt)*) => {
        return Err($crate::error::err!($kind, $($arg)*))
    };
}

pub(crate) use bail;
pub(crate) use err;
