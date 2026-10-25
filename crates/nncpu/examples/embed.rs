//! Semantic search with a text-embedding model.
//!
//!     cargo run --release --example embed -- embeddinggemma-300M-Q8_0.gguf "Which planet is known as the Red Planet?"
//!
//! Embeds a few documents and the query, then ranks the documents by how
//! close their meaning is to the query's. Embeddings have unit length, so
//! the dot product of two is their cosine similarity.

use nncpu::Model;
use std::process::ExitCode;

const DOCUMENTS: [&str; 4] = [
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
];

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: {} EMBEDDING-MODEL.gguf \"query\"", args[0]);
        return ExitCode::from(2);
    }
    let model = match Model::load(&args[1]) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("cannot load {}: {e}", args[1]);
            return ExitCode::FAILURE;
        }
    };
    if model.info().n_embd_out == 0 {
        eprintln!("{} is not a text-embedding model", model.name());
        return ExitCode::FAILURE;
    }
    match rank(&model, &args[2]) {
        Ok(ranked) => {
            for (score, doc) in ranked {
                println!("{score:.3}  {doc}");
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

/// The documents, best match first. EmbeddingGemma wants the task in front
/// of the text: one prefix for search queries, another for the documents
/// they are matched against.
fn rank(model: &Model, query: &str) -> nncpu::Result<Vec<(f32, &'static str)>> {
    let q = model.embedding(&format!("task: search result | query: {query}"))?;
    let mut ranked = Vec::new();
    for doc in DOCUMENTS {
        let d = model.embedding(&format!("title: none | text: {doc}"))?;
        let score: f32 = q.iter().zip(&d).map(|(a, b)| a * b).sum();
        ranked.push((score, doc));
    }
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
    Ok(ranked)
}
