//! Downloads through the system's curl (on Windows 10 and later, and nearly
//! every Linux), or wget where curl is missing outside Windows: following
//! redirects, resuming a partial file, and reporting progress by watching
//! the file grow.

use std::fmt;
use std::fs;
use std::io::{self, Read};
use std::path::Path;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::thread;
use std::time::Duration;

/// How often the download's progress is looked at.
const POLL: Duration = Duration::from_millis(200);

#[derive(Debug, PartialEq, Eq)]
pub enum DownloadError {
    /// The progress callback asked to stop, or Ctrl-C ended the downloader.
    Cancelled,
    /// Why it failed, e.g. "the download failed (curl: (6) Could not resolve
    /// host: huggingface.co)".
    Failed(String),
}

impl fmt::Display for DownloadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DownloadError::Cancelled => f.write_str("cancelled"),
            DownloadError::Failed(why) => f.write_str(why),
        }
    }
}

/// The size of the file at `path`, or 0.
pub fn file_size(path: &Path) -> u64 {
    fs::metadata(path).map_or(0, |m| m.len())
}

/// Whether the file at `path` starts with the GGUF magic.
pub fn is_gguf(path: &Path) -> bool {
    let mut magic = [0u8; 4];
    fs::File::open(path).and_then(|mut f| f.read_exact(&mut magic)).is_ok() && &magic == b"GGUF"
}

/// Whether a download of `bytes` bytes arrived intact at `path`: exactly
/// that long, and a GGUF file.
pub fn arrived_intact(path: &Path, bytes: u64) -> bool {
    file_size(path) == bytes && is_gguf(path)
}

/// Move `source` over `destination`, replacing it; on one volume, atomically.
pub fn replace_file(source: &Path, destination: &Path) -> io::Result<()> {
    fs::rename(source, destination)
}

/// Fetch `url` into the file `dest`, following redirects. A partial `dest`
/// is resumed where the server allows. `progress(done, total)` is told the
/// size of `dest` about five times a second, `total` being the size to
/// expect; it returns false to stop the download.
pub fn download(
    url: &str,
    dest: &Path,
    total: u64,
    mut progress: impl FnMut(u64, u64) -> bool,
) -> Result<(), DownloadError> {
    let mut child = spawn_downloader(url, dest)?;
    // The downloader's own error message, read as it comes so that a chatty
    // downloader can never fill the pipe and stall.
    let messages = child.stderr.take().map(|mut pipe| {
        thread::spawn(move || {
            let mut text = Vec::new();
            pipe.read_to_end(&mut text).ok();
            text
        })
    });
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {}
            Err(e) => {
                stop(&mut child);
                return Err(DownloadError::Failed(format!("lost track of the downloader: {e}")));
            }
        }
        if !progress(file_size(dest), total) {
            stop(&mut child);
            return Err(DownloadError::Cancelled);
        }
        thread::sleep(POLL);
    };
    // Ctrl-C reaches the downloader as well as nncpu, and may end it first.
    if !progress(file_size(dest), total) || ended_by_ctrl_c(status) {
        return Err(DownloadError::Cancelled);
    }
    if status.success() {
        return Ok(());
    }
    let messages = messages.and_then(|reader| reader.join().ok()).unwrap_or_default();
    let messages = String::from_utf8_lossy(&messages);
    let detail = messages.lines().rev().map(str::trim).find(|line| !line.is_empty());
    Err(DownloadError::Failed(match (detail, status.code()) {
        (Some(message), _) => format!("the download failed ({message})"),
        (None, Some(code)) => format!("the download failed (downloader exit code {code})"),
        (None, None) => "the download was interrupted".into(),
    }))
}

fn spawn_downloader(url: &str, dest: &Path) -> Result<Child, DownloadError> {
    let run = |program: &str, options: &[&str]| {
        Command::new(program)
            .args(options)
            .arg(dest)
            .arg(url)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
    };
    let curl = [
        "--fail",
        "--location",
        "--silent",
        "--show-error",
        "--retry",
        "3",
        "--continue-at",
        "-",
        "--output",
    ];
    let mut spawned = run("curl", &curl).map_err(|e| ("curl", e));
    if !cfg!(windows) && matches!(&spawned, Err((_, e)) if e.kind() == io::ErrorKind::NotFound) {
        spawned = run("wget", &["--quiet", "--continue", "--output-document"]).map_err(|e| ("wget", e));
    }
    spawned.map_err(|(program, e)| {
        DownloadError::Failed(if e.kind() != io::ErrorKind::NotFound {
            format!("cannot start {program}: {e}")
        } else if cfg!(windows) {
            "downloading needs curl, which comes with Windows 10 and later; install it, or fetch the file yourself"
                .into()
        } else {
            "downloading needs curl or wget; install one, or fetch the file yourself".into()
        })
    })
}

fn stop(child: &mut Child) {
    child.kill().ok();
    child.wait().ok();
}

/// Whether the downloader ended because of Ctrl-C.
#[cfg(windows)]
fn ended_by_ctrl_c(status: ExitStatus) -> bool {
    status.code() == Some(0xC000_013A_u32 as i32) // STATUS_CONTROL_C_EXIT
}

#[cfg(unix)]
fn ended_by_ctrl_c(status: ExitStatus) -> bool {
    use std::os::unix::process::ExitStatusExt;
    status.signal() == Some(2) // SIGINT
}

#[cfg(not(any(windows, unix)))]
fn ended_by_ctrl_c(_: ExitStatus) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_dir;
    use std::path::PathBuf;

    fn gguf_bytes(n: usize) -> Vec<u8> {
        let mut bytes: Vec<u8> = (0..n).map(|i| (i % 251) as u8).collect();
        bytes[..4].copy_from_slice(b"GGUF");
        bytes
    }

    #[test]
    fn checks_size_and_magic() {
        let dir = test_dir("intact");
        let good = dir.join("good.gguf");
        fs::write(&good, gguf_bytes(1000)).unwrap();
        assert!(arrived_intact(&good, 1000));
        assert!(!arrived_intact(&good, 999), "one byte short of the catalogue's size");
        assert!(!arrived_intact(&good, 1001));
        let html = dir.join("error-page.gguf");
        fs::write(&html, b"<html>not found</html>").unwrap();
        assert!(!arrived_intact(&html, 22));
        let tiny = dir.join("tiny.gguf");
        fs::write(&tiny, b"GG").unwrap();
        assert!(!is_gguf(&tiny));
        assert!(!arrived_intact(&dir.join("missing.gguf"), 0));
        assert_eq!(file_size(&dir.join("missing.gguf")), 0);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn replaces_an_existing_file() {
        let dir = test_dir("replace");
        let (source, destination) = (dir.join("model.gguf.part"), dir.join("model.gguf"));
        fs::write(&source, "verified replacement").unwrap();
        fs::write(&destination, "old partial download").unwrap();
        replace_file(&source, &destination).unwrap();
        assert_eq!(fs::read_to_string(&destination).unwrap(), "verified replacement");
        assert!(!source.exists());
        fs::remove_dir_all(&dir).unwrap();
    }

    fn curl_available() -> bool {
        Command::new("curl")
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok()
    }

    fn file_url(path: &Path) -> String {
        let path = path.canonicalize().unwrap().display().to_string();
        let path = path.strip_prefix(r"\\?\").unwrap_or(&path).replace('\\', "/");
        if path.starts_with('/') {
            format!("file://{path}")
        } else {
            format!("file:///{path}")
        }
    }

    /// A real transfer through curl, from a file:// URL: a fresh download, a
    /// resumed one, a cancelled one and a failing one.
    #[test]
    fn downloads_and_resumes_with_curl() {
        if !curl_available() {
            eprintln!("skipped: no curl");
            return;
        }
        let dir = test_dir("curl");
        let source = dir.join("source.gguf");
        let data = gguf_bytes(300_000);
        fs::write(&source, &data).unwrap();
        let url = file_url(&source);

        let fresh: PathBuf = dir.join("fresh.gguf.part");
        let mut calls = 0;
        download(&url, &fresh, data.len() as u64, |_, total| {
            assert_eq!(total, 300_000);
            calls += 1;
            true
        })
        .unwrap();
        assert!(calls >= 1);
        assert_eq!(fs::read(&fresh).unwrap(), data);

        let resumed = dir.join("resumed.gguf.part");
        fs::write(&resumed, &data[..123_456]).unwrap();
        download(&url, &resumed, data.len() as u64, |_, _| true).unwrap();
        assert_eq!(fs::read(&resumed).unwrap(), data);

        let cancelled = dir.join("cancelled.gguf.part");
        assert_eq!(
            download(&url, &cancelled, data.len() as u64, |_, _| false),
            Err(DownloadError::Cancelled)
        );

        let missing = file_url(&dir).trim_end_matches('/').to_string() + "/no-such-file.gguf";
        match download(&missing, &dir.join("missing.gguf.part"), 1, |_, _| true) {
            Err(DownloadError::Failed(why)) => assert!(why.starts_with("the download failed (curl: (37)"), "{why}"),
            other => panic!("expected a failure, got {other:?}"),
        }
        fs::remove_dir_all(&dir).unwrap();
    }
}
