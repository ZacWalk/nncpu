//! The smallest complete nncpu program.
//!
//!     cargo run --release --example minimal -- MODEL.gguf "Why is the sky blue?"
//!
//! Loads a model, sends one message in the model's own chat format, and
//! streams the reply to stdout.

use nncpu::{GenerateParams, Model, Piece, Session, SessionParams};
use std::io::Write;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: {} MODEL.gguf \"message\"", args[0]);
        return ExitCode::from(2);
    }
    if !nncpu::cpu_supported() {
        eprintln!("note: this CPU lacks AVX2 and FMA, so nncpu runs its much slower portable kernels");
    }

    // The defaults: 8-bit weights, no progress callback.
    let model = match Model::load(&args[1]) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("cannot load {}: {e}", args[1]);
            return ExitCode::FAILURE;
        }
    };
    let mut session = match Session::new(&model, &SessionParams::default()) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("cannot start a session: {e}");
            return ExitCode::FAILURE;
        }
    };

    let result = session.chat(&args[2], &GenerateParams::default(), |piece| {
        if let Piece::Text(text) = piece {
            print!("{text}");
            std::io::stdout().flush().ok();
        }
        true // returning false would stop the reply
    });
    println!();
    match result {
        Ok(r) => {
            let secs = r.decode.as_secs_f64();
            let rate = if secs > 0.0 { r.n_generated as f64 / secs } else { 0.0 };
            eprintln!("[{} tokens at {rate:.1} tokens/s]", r.n_generated);
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
