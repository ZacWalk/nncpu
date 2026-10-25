//! Finding, choosing and downloading models.

use crate::catalog::{CATALOG, CatalogEntry, find_catalog_entry};
use crate::cli;
use crate::download::{self, DownloadError, file_size};
use crate::platform::{self, InterruptScope};
use crate::ui::{self, ProgressBar, format_bytes, format_duration, outln, tidy_path, ui_error, ui_info};
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// The file name of `path`, for display and matching.
pub fn file_name(path: &Path) -> String {
    path.file_name().map_or_else(
        || path.display().to_string(),
        |name| name.to_string_lossy().into_owned(),
    )
}

/// Drafters only help another model generate: mtp-* (which nncpu uses with
/// Gemma 4) and dflash-* (which it does not run).
fn is_drafter_file(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    name.starts_with("mtp-") || name.starts_with("dflash-")
}

/// "q8_0", "q4_k_m", "iq4_xs", "bf16", "f16", "f32", "ud": a file name's
/// quantization part (lower case).
fn is_quant_label(s: &str) -> bool {
    if matches!(s, "bf16" | "f16" | "f32" | "ud") {
        return true;
    }
    let rest = s.strip_prefix("iq").or_else(|| s.strip_prefix('q'));
    rest.is_some_and(|rest| rest.starts_with(|c: char| c.is_ascii_digit()))
}

/// Whether `drafter` (a file name, mtp-<name>[-<quant>].gguf) is made for
/// the model file `model`: its file name starts with <name>, then '-' or '.'.
fn drafter_fits(drafter: &str, model: &str) -> bool {
    let drafter = drafter.to_ascii_lowercase();
    let target = model.to_ascii_lowercase();
    let Some(mut stem) = drafter.strip_prefix("mtp-").and_then(|d| d.strip_suffix(".gguf")) else {
        return false;
    };
    while let Some(dash) = stem.rfind('-')
        && is_quant_label(&stem[dash + 1..])
    {
        stem = &stem[..dash];
    }
    !stem.is_empty()
        && target.len() > stem.len()
        && target.starts_with(stem)
        && matches!(target.as_bytes()[stem.len()], b'-' | b'.')
}

/// Where models are looked for: ./models, ~/.lmstudio/models (LM Studio's
/// downloads) and ~/models.
fn search_dirs() -> Vec<PathBuf> {
    let mut dirs = vec![PathBuf::from("models")];
    if let Some(home) = platform::home_dir() {
        dirs.push(home.join(".lmstudio").join("models"));
        dirs.push(home.join("models"));
    }
    dirs
}

/// Every file under `root`, sorted. Links to files count; linked folders are
/// not followed.
fn files_under(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut dirs = vec![root.to_path_buf()];
    while let Some(dir) = dirs.pop() {
        let Ok(entries) = fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            let path = entry.path();
            match entry.file_type() {
                Ok(t) if t.is_dir() => dirs.push(path),
                Ok(t) if t.is_file() || (t.is_symlink() && path.is_file()) => files.push(path),
                _ => {}
            }
        }
    }
    // By the bytes of the path, as a sorted list of path strings would be.
    files.sort_by(|a, b| a.as_os_str().cmp(b.as_os_str()));
    files
}

/// .gguf files under ./models, ~/.lmstudio/models and ~/models that can be
/// chatted with or embedded with: without vision projectors (mmproj-*) and
/// drafters (mtp-*, dflash-*), each found once.
pub fn find_models() -> Vec<PathBuf> {
    find_models_in(&search_dirs())
}

fn find_models_in(dirs: &[PathBuf]) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut seen = HashSet::new();
    for dir in dirs.iter().filter(|d| d.is_dir()) {
        for path in files_under(dir) {
            let name = file_name(&path);
            let lower = name.to_ascii_lowercase();
            if !lower.ends_with(".gguf") || lower.contains("mmproj") || is_drafter_file(&name) {
                continue;
            }
            // The same file may be reachable from two of the folders.
            if seen.insert(fs::canonicalize(&path).unwrap_or_else(|_| path.clone())) {
                found.push(path);
            }
        }
    }
    found
}

/// A multi-token-prediction drafter for the model at `model_path`: a file
/// mtp-<name>[-<quant>].gguf, next to the model or in the model folders,
/// where the model's file name starts with <name> ("gemma-4-E2B-it" for
/// gemma-4-E2B-it-Q4_K_M.gguf).
pub fn find_draft_model(model_path: &Path) -> Option<PathBuf> {
    let beside = match model_path.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir.to_path_buf(),
        _ => PathBuf::from("."),
    };
    let mut dirs = vec![beside];
    dirs.extend(search_dirs());
    find_draft_model_in(model_path, &dirs)
}

fn find_draft_model_in(model_path: &Path, dirs: &[PathBuf]) -> Option<PathBuf> {
    let target = file_name(model_path);
    dirs.iter()
        .filter(|d| d.is_dir())
        .flat_map(|d| files_under(d))
        .find(|f| drafter_fits(&file_name(f), &target))
}

/// The catalogue entry of a drafter for the model at `model_path`.
pub fn find_catalog_drafter(model_path: &Path) -> Option<&'static CatalogEntry> {
    let target = file_name(model_path);
    CATALOG.iter().find(|e| drafter_fits(e.file, &target))
}

/// Where downloads go: ./models if that folder exists, else ~/models.
pub fn download_dir() -> PathBuf {
    download_dir_for(Path::new("models").is_dir(), platform::home_dir())
}

fn download_dir_for(local_models: bool, home: Option<PathBuf>) -> PathBuf {
    match home {
        Some(home) if !local_models => home.join("models"),
        _ => PathBuf::from("models"),
    }
}

fn absolute(path: &Path) -> PathBuf {
    std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf())
}

/// An answer to a menu.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Choice {
    Number(usize),
    Letter(char),
    EndOfInput,
}

/// Ask for a menu choice until one is given: a number in [1, count], a blank
/// line for `fallback`, or one of `letters`.
fn read_choice(prompt: &str, count: usize, fallback: usize, letters: &str) -> Choice {
    loop {
        eprint!("{prompt}");
        let Some(line) = platform::read_line() else {
            return Choice::EndOfInput;
        };
        if let Some(choice) = parse_choice(&line, count, fallback, letters) {
            return choice;
        }
        eprintln!(
            "Please type a number from 1 to {count}{}.",
            if letters.is_empty() { "" } else { ", or a letter shown" }
        );
    }
}

fn parse_choice(line: &str, count: usize, fallback: usize, letters: &str) -> Option<Choice> {
    let s = line.trim_matches([' ', '\t']);
    if s.is_empty() {
        return Some(Choice::Number(fallback));
    }
    if let [c] = s.as_bytes() {
        let c = char::from(c | 0x20); // either case
        if letters.contains(c) {
            return Some(Choice::Letter(c));
        }
    }
    let n = cli::parse_long(s).and_then(|n| usize::try_from(n).ok())?;
    (1..=count).contains(&n).then_some(Choice::Number(n))
}

/// Download `e` to download_dir(), asking first unless `yes`. The path, or
/// None after telling the user why not.
fn download_model(e: &CatalogEntry, yes: bool) -> Option<PathBuf> {
    let dir = download_dir();
    let path = dir.join(e.file);
    if file_size(&path) == e.bytes {
        return Some(path);
    }

    if !yes {
        if !platform::stdin_is_terminal() {
            ui_error!(
                "{} is not downloaded; run nncpu --download {}, or add --yes",
                e.name,
                e.name
            );
            return None;
        }
        let question = format!(
            "Download {} ({}) to {}?",
            e.file,
            format_bytes(e.bytes),
            tidy_path(&absolute(&dir))
        );
        if !ui::confirm(&question, true) {
            return None;
        }
    }

    if let Err(err) = fs::create_dir_all(&dir) {
        ui_error!("cannot create '{}': {err}", dir.display());
        return None;
    }

    let mut part = path.clone().into_os_string();
    part.push(".part");
    let part = PathBuf::from(part);
    if file_size(&part) > e.bytes {
        fs::remove_file(&part).ok(); // not ours to resume
    }
    if file_size(&part) != e.bytes {
        let mut bar = ProgressBar::new(format!("Downloading {}", e.file));
        let mut speed = Speed::default();
        let result = {
            let _scope = InterruptScope::begin(); // Ctrl-C stops the download instead of ending nncpu
            download::download(e.url, &part, e.bytes, |done, total| {
                if platform::interrupted() {
                    return false;
                }
                let (fraction, detail) = speed.progress(done, total);
                bar.update(fraction, &detail);
                true
            })
        };
        match result {
            Ok(()) => bar.finish("done"),
            Err(err) => {
                bar.finish("stopped");
                match err {
                    DownloadError::Cancelled => ui_info!("Download stopped. Run the same command again to resume it."),
                    DownloadError::Failed(why) => ui_error!("{why}. Run the same command again to resume."),
                }
                return None;
            }
        }
    }

    if !download::arrived_intact(&part, e.bytes) {
        ui_error!(
            "{} did not arrive intact (expected {} bytes); delete '{}' and try again",
            e.file,
            e.bytes,
            tidy_path(&part)
        );
        return None;
    }
    if let Err(err) = download::replace_file(&part, &path) {
        ui_error!("cannot replace '{}': {err}", path.display());
        return None;
    }
    ui_info!("Saved {}", tidy_path(&absolute(&path)));
    Some(path)
}

/// A download's rate and time left, measured from the first report.
#[derive(Default)]
struct Speed {
    start: Option<(Instant, u64)>,
}

impl Speed {
    /// The fraction done, and the bar's detail: "456 MB / 1013 MB  12 MB/s  0:45 left".
    fn progress(&mut self, done: u64, total: u64) -> (f64, String) {
        let now = Instant::now();
        let (start, start_bytes) = *self.start.get_or_insert((now, done));
        let secs = now.duration_since(start).as_secs_f64();
        let rate = if secs > 0.5 {
            done.saturating_sub(start_bytes) as f64 / secs
        } else {
            0.0
        };
        let mut detail = format!(
            "{} / {}",
            format_bytes(done),
            if total > 0 { format_bytes(total) } else { "?".into() }
        );
        if rate > 0.0 {
            detail += &format!("  {}/s", format_bytes(rate as u64));
            if total > done {
                detail += &format!("  {} left", format_duration((total - done) as f64 / rate));
            }
        }
        (if total > 0 { done as f64 / total as f64 } else { 0.0 }, detail)
    }
}

/// Offer the catalogue (without drafters, which cannot run on their own).
/// With `yes`, takes the recommended model.
fn choose_download(yes: bool) -> Option<PathBuf> {
    if yes {
        return download_model(&CATALOG[0], true);
    }
    let menu: Vec<&CatalogEntry> = CATALOG.iter().filter(|e| !is_drafter_file(e.file)).collect();
    eprintln!("\nModels nncpu can download:");
    for (i, e) in menu.iter().enumerate() {
        eprintln!(
            "  {}) {:<34} {:>8}  {}{}{}",
            i + 1,
            e.file,
            format_bytes(e.bytes),
            ui::dim(),
            e.description,
            ui::reset()
        );
    }
    eprintln!("Downloads are saved to {}.", tidy_path(&absolute(&download_dir())));
    match read_choice("Choose a model [1], or q to quit: ", menu.len(), 1, "q") {
        Choice::Number(n) => download_model(menu[n - 1], true),
        _ => None,
    }
}

/// The model to run: `requested` (-m) if given — a path, or a catalogue name
/// that is downloaded if missing — else the one model found, else a choice
/// between the models found, else an offer to download one. `yes` answers
/// download questions. None after telling the user why.
pub fn resolve_model(requested: &str, yes: bool) -> Option<PathBuf> {
    if !requested.is_empty() {
        if Path::new(requested).is_file() {
            return Some(PathBuf::from(requested));
        }
        if let Some(e) = find_catalog_entry(requested) {
            let found = find_models()
                .into_iter()
                .find(|p| file_name(p) == e.file && file_size(p) == e.bytes);
            return found.or_else(|| download_model(e, yes));
        }
        ui_error!("there is no model file '{requested}'. Run nncpu --download to see the models nncpu can fetch.");
        return None;
    }

    let models = find_models();
    let interactive = platform::stdin_is_terminal();
    if models.is_empty() {
        ui_info!("No models found in ./models, ~/.lmstudio/models or ~/models.");
        if !interactive && !yes {
            ui_error!("pass -m FILE, or run nncpu --download to fetch a model");
            return None;
        }
        return choose_download(yes);
    }
    if models.len() == 1 {
        return models.into_iter().next();
    }
    if !interactive {
        ui_error!(
            "{} models found; choose one with -m FILE (nncpu --list-models shows them)",
            models.len()
        );
        return None;
    }

    eprintln!("Models found:");
    for (i, m) in models.iter().enumerate() {
        eprintln!(
            "  {}) {:<40} {:>8}  {}{}{}",
            i + 1,
            file_name(m),
            format_bytes(file_size(m)),
            ui::dim(),
            tidy_path(m.parent().unwrap_or(Path::new(""))),
            ui::reset()
        );
    }
    eprintln!("  d) download another model");
    match read_choice("Choose a model [1]: ", models.len(), 1, "dq") {
        Choice::Letter('d') => choose_download(false),
        Choice::Number(n) => Some(models[n - 1].clone()),
        _ => None,
    }
}

/// --list-models
pub fn run_list_models() -> u8 {
    let models = find_models();
    if models.is_empty() {
        ui_info!("No models found in ./models, ~/.lmstudio/models or ~/models. Run nncpu --download to fetch one.");
        return 1;
    }
    for m in &models {
        outln!("{:>8}  {}", format_bytes(file_size(m)), m.display());
    }
    0
}

/// --download [NAME]
pub fn run_download(name: &str, yes: bool) -> u8 {
    let path = if name.is_empty() {
        choose_download(yes)
    } else {
        let Some(e) = find_catalog_entry(name) else {
            ui_error!("nncpu does not know a model called '{name}'. These are the ones it can download:");
            for c in CATALOG {
                ui_info!("  {:<18} {}", c.name, c.description);
            }
            return 1;
        };
        download_model(e, yes)
    };
    match path {
        Some(path) => {
            outln!("{}", path.display());
            0
        }
        None => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_dir;

    #[test]
    fn quant_labels() {
        for label in [
            "q8_0", "q4_k_m", "q6_k", "iq4_xs", "iq2_xxs", "bf16", "f16", "f32", "ud", "q4",
        ] {
            assert!(is_quant_label(label), "{label}");
        }
        for word in ["it", "e2b", "a4b", "4b", "q", "iq", "qwen3", "instruct", ""] {
            assert!(!is_quant_label(word), "{word}");
        }
    }

    #[test]
    fn drafters_fit_their_models() {
        assert!(drafter_fits(
            "mtp-gemma-4-E2B-it-Q8_0.gguf",
            "gemma-4-E2B-it-Q4_K_M.gguf"
        ));
        assert!(drafter_fits("mtp-gemma-4-E2B-it-Q8_0.gguf", "gemma-4-E2B-it-BF16.gguf"));
        assert!(drafter_fits("MTP-gemma-4-e2b-it.gguf", "gemma-4-E2B-it.Q4_K_M.gguf"));
        assert!(drafter_fits(
            "mtp-gemma-4-26B-A4B-it-Q8_0.gguf",
            "gemma-4-26B-A4B-it-Q4_K_M.gguf"
        ));
        assert!(drafter_fits(
            "mtp-gemma-4-E4B-it-UD-Q8_K_XL.gguf",
            "gemma-4-E4B-it-UD-Q4_K_XL.gguf"
        ));
        // E2B's drafter is not E4B's, and the model must go on past the name.
        assert!(!drafter_fits(
            "mtp-gemma-4-E2B-it-Q8_0.gguf",
            "gemma-4-E4B-it-Q4_K_M.gguf"
        ));
        assert!(!drafter_fits(
            "mtp-gemma-4-E2B-it-Q8_0.gguf",
            "gemma-4-E2B-itx-Q4_K_M.gguf"
        ));
        assert!(!drafter_fits("mtp-gemma-4-E2B-it-Q8_0.gguf", "gemma-4-E2B-it"));
        assert!(!drafter_fits(
            "dflash-gemma-4-E2B-it.gguf",
            "gemma-4-E2B-it-Q4_K_M.gguf"
        ));
        assert!(!drafter_fits("mtp-.gguf", "x.gguf"));
        assert!(
            !drafter_fits("mtp--Q8_0.gguf", "-model.gguf"),
            "no name is left once the quantization goes"
        );
        assert!(!drafter_fits(
            "mtp-gemma-4-E2B-it-Q8_0.bin",
            "gemma-4-E2B-it-Q4_K_M.gguf"
        ));
    }

    #[test]
    fn catalogue_drafters() {
        let drafter = |model: &str| find_catalog_drafter(Path::new(model)).map(|e| e.name);
        assert_eq!(drafter("models/gemma-4-E2B-it-Q4_K_M.gguf"), Some("mtp-gemma-4-e2b-q8"));
        assert_eq!(drafter("gemma-4-E4B-it-BF16.gguf"), Some("mtp-gemma-4-e4b-q8"));
        assert_eq!(
            drafter("/m/gemma-4-26B-A4B-it-Q4_K_M.gguf"),
            Some("mtp-gemma-4-26b-a4b-q8")
        );
        assert_eq!(drafter("gemma-3-1b-it-Q6_K.gguf"), None);
        assert_eq!(drafter("Qwen3.5-0.8B-Q8_0.gguf"), None);
    }

    #[test]
    fn drafter_files_are_recognised() {
        assert!(is_drafter_file("mtp-gemma-4-E2B-it-Q8_0.gguf"));
        assert!(is_drafter_file("MTP-x.gguf"));
        assert!(is_drafter_file("dflash-qwen.gguf"));
        assert!(!is_drafter_file("gemma-mtp-x.gguf"));
    }

    #[test]
    fn menu_choices() {
        assert_eq!(parse_choice("", 3, 1, "q"), Some(Choice::Number(1)));
        assert_eq!(parse_choice(" \t", 3, 1, "q"), Some(Choice::Number(1)));
        assert_eq!(parse_choice("2", 3, 1, "q"), Some(Choice::Number(2)));
        assert_eq!(parse_choice(" 3 ", 3, 1, "q"), Some(Choice::Number(3)));
        assert_eq!(parse_choice("+3", 3, 1, "q"), Some(Choice::Number(3)));
        assert_eq!(parse_choice("4", 3, 1, "q"), None);
        assert_eq!(parse_choice("0", 3, 1, "q"), None);
        assert_eq!(parse_choice("-1", 3, 1, "q"), None);
        assert_eq!(parse_choice("q", 3, 1, "q"), Some(Choice::Letter('q')));
        assert_eq!(parse_choice("Q", 3, 1, "q"), Some(Choice::Letter('q')));
        assert_eq!(parse_choice("D", 3, 1, "dq"), Some(Choice::Letter('d')));
        assert_eq!(parse_choice("d", 3, 1, "q"), None);
        assert_eq!(parse_choice("quit", 3, 1, "q"), None);
        assert_eq!(parse_choice("2x", 3, 1, "q"), None);
    }

    #[test]
    fn download_folder() {
        let home = PathBuf::from("home");
        assert_eq!(download_dir_for(true, Some(home.clone())), PathBuf::from("models"));
        assert_eq!(download_dir_for(false, Some(home.clone())), home.join("models"));
        assert_eq!(download_dir_for(false, None), PathBuf::from("models"));
    }

    #[test]
    fn finds_models_once_without_projectors_or_drafters() {
        let root = test_dir("find-models");
        let (a, b) = (root.join("a"), root.join("b"));
        for (dir, files) in [
            (
                &a,
                &[
                    "z.gguf",
                    "gemma-3-1b-it-Q6_K.gguf",
                    "mmproj-gemma.gguf",
                    "notes.txt",
                    "x.GGUF",
                ][..],
            ),
            (
                &a.join("sub").join("deeper"),
                &["nested.gguf", "mtp-gemma-4-E2B-it-Q8_0.gguf"][..],
            ),
            (&b, &["other.gguf", "dflash-x.gguf"][..]),
        ] {
            fs::create_dir_all(dir).unwrap();
            for f in files {
                fs::write(dir.join(f), b"GGUF").unwrap();
            }
        }
        // `a` twice, as when ./models is ~/models, and a folder that is not there.
        let found = find_models_in(&[a.clone(), root.join("missing"), b.clone(), a.clone()]);
        let names: Vec<String> = found
            .iter()
            .map(|p| p.strip_prefix(&root).unwrap().display().to_string())
            .collect();
        let sep = std::path::MAIN_SEPARATOR;
        assert_eq!(
            names,
            [
                format!("a{sep}gemma-3-1b-it-Q6_K.gguf"),
                format!("a{sep}sub{sep}deeper{sep}nested.gguf"),
                format!("a{sep}x.GGUF"),
                format!("a{sep}z.gguf"),
                format!("b{sep}other.gguf"),
            ]
        );

        // The drafter is found beside the model or in the folders, wherever it nests.
        let model = a.join("gemma-4-E2B-it-Q4_K_M.gguf");
        let drafter = find_draft_model_in(&model, &[b.clone(), a.clone()]).unwrap();
        assert_eq!(file_name(&drafter), "mtp-gemma-4-E2B-it-Q8_0.gguf");
        assert_eq!(
            find_draft_model_in(&a.join("gemma-3-1b-it-Q6_K.gguf"), &[a.clone(), b.clone()]),
            None
        );
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn download_progress_detail() {
        let mut speed = Speed::default();
        let (fraction, detail) = speed.progress(512 << 20, 1 << 30);
        assert_eq!(fraction, 0.5);
        assert_eq!(detail, "512 MB / 1.0 GB", "no rate before half a second has passed");
        let (fraction, detail) = speed.progress(0, 0);
        assert_eq!((fraction, detail.as_str()), (0.0, "0 B / ?"));
    }
}
