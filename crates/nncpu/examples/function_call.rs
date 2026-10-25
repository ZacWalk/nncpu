//! A function-calling round trip with FunctionGemma.
//!
//!     cargo run --release --example function_call -- functiongemma-270m-it-q8_0.gguf "What's the weather in Tokyo?"
//!
//! Declares one function in the system prompt (FunctionGemma's developer
//! turn), lets the model call it, hands back a made-up result in the same
//! turn, and lets the model answer in words.

use nncpu::{GenerateParams, Model, Piece, SamplerParams, Session, SessionParams};
use std::process::ExitCode;

const DECLARATIONS: &str = "You are a model that can do function calling with the following functions\
    <start_function_declaration>declaration:get_current_weather{description:<escape>Gets the current weather \
    in a given location.<escape>,parameters:{properties:{location:{description:<escape>The city and state, \
    e.g. \"San Francisco, CA\" or \"Tokyo, JP\"<escape>,type:<escape>STRING<escape>}},\
    required:[<escape>location<escape>],type:<escape>OBJECT<escape>}}<end_function_declaration>";

/// What the function returned, in FunctionGemma's syntax.
const RESULT: &str =
    "response:get_current_weather{temperature:15,weather:<escape>sunny<escape>}<end_function_response>";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: {} FUNCTIONGEMMA.gguf \"question\"", args[0]);
        return ExitCode::from(2);
    }
    match run(&args[1], &args[2]) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(path: &str, question: &str) -> nncpu::Result<()> {
    let model = Model::load(path)?;
    let session_params = SessionParams {
        system_prompt: DECLARATIONS.into(),
        ..Default::default()
    };
    let mut session = Session::new(&model, &session_params)?;
    // Greedy, so the run is repeatable.
    let params = GenerateParams {
        sampler: SamplerParams::greedy(),
        ..Default::default()
    };

    let mut reply = String::new();
    let mut collect = |piece: Piece| {
        if let Piece::Text(text) = piece {
            reply.push_str(text);
        }
        true
    };
    session.chat(question, &params, &mut collect)?;
    println!("model:    {reply}");
    if !reply.contains("<start_function_call>call:get_current_weather") {
        println!("(the model answered without calling the function)");
        return Ok(());
    }

    // Generation stopped where the function's response belongs. Run the
    // function (here, pretend) and continue the same turn with its result.
    println!("function: {RESULT}");
    let tokens = model.tokenize(RESULT, false);
    reply.clear();
    session.generate(&tokens, &params, |piece| {
        if let Piece::Text(text) = piece {
            reply.push_str(text);
        }
        true
    })?;
    println!("model:    {reply}");
    Ok(())
}
