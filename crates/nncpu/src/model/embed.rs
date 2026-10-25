//! Attention spans, pooling and text embeddings.

use super::forward::model_forward;
use super::kv::kv_init;
use super::{EMBED_MAX_TOKENS, ModelData, Pooling, model_gemv};
use crate::error::{Result, bail};

/// The keys that the query at `pos` attends to, `(first, last)` inclusive,
/// in a sequence whose keys fill [0, n_seq). A causal model sees its prefix
/// up to `pos` (an MTP drafter's query, at the position after the keys, sees
/// them all), and a sliding layer (window > 0) only the last `window`
/// positions up to `pos`. A bidirectional model sees the whole sequence, and
/// a sliding layer only |key - pos| <= window / 2.
pub(crate) fn attn_span(causal: bool, window: usize, pos: usize, n_seq: usize) -> (usize, usize) {
    let end = n_seq.saturating_sub(1);
    if causal {
        let first = if window > 0 && pos + 1 > window {
            pos + 1 - window
        } else {
            0
        };
        return (first, pos.min(end));
    }
    if window > 0 {
        let half = window / 2;
        return (pos.saturating_sub(half), end.min(pos + half));
    }
    (0, end)
}

/// Pool `n_tok` rows of `n` floats (token-major) into `out`, as `pooling` says.
fn pool_rows(pooling: Pooling, rows: &[f32], n_tok: usize, n: usize, out: &mut [f32]) {
    if matches!(pooling, Pooling::Cls | Pooling::Last) {
        let row = if pooling == Pooling::Cls { 0 } else { n_tok - 1 };
        out[..n].copy_from_slice(&rows[row * n..(row + 1) * n]);
        return;
    }
    for (i, o) in out[..n].iter_mut().enumerate() {
        let sum: f64 = (0..n_tok).map(|t| rows[t * n + i] as f64).sum();
        *o = (sum / n_tok as f64) as f32;
    }
}

/// Scale `x` to unit Euclidean length (left as it is when it is all zero).
fn l2_normalize(x: &mut [f32]) {
    let ss: f64 = x.iter().map(|&v| v as f64 * v as f64).sum();
    if ss.is_nan() || ss <= 0.0 {
        return;
    }
    let inv = (1.0 / ss.sqrt()) as f32;
    for v in x {
        *v *= inv;
    }
}

/// An embedding model's vector for `text`: the text between the model's BOS
/// and EOS (cut to fit its context), run through the model with
/// bidirectional attention, pooled, projected, cut to its first `out.len()`
/// values and scaled to unit length. Allocates its own context.
pub(crate) fn model_embed(m: &ModelData, text: &str, out: &mut [f32]) -> Result<()> {
    let h = &m.hparams;
    let n_out = out.len();
    if h.n_embd_out == 0 {
        bail!(
            UnsupportedModel,
            "{} generates text; it is not a text-embedding model",
            m.name
        );
    }
    if n_out == 0 || n_out > h.n_embd_out {
        bail!(
            InvalidArgument,
            "this model's embeddings have up to {} values, not {n_out}",
            h.n_embd_out
        );
    }

    // [BOS] text [EOS], with the text cut so that the whole fits.
    let mut ids = m.vocab.tokenize(text, m.vocab.add_bos);
    let eos = m.vocab.add_eos && m.vocab.eos_id >= 0;
    let limit = h.n_ctx.min(EMBED_MAX_TOKENS) - usize::from(eos);
    if ids.len() > limit {
        log::debug!("embedding the first {limit} of {} tokens", ids.len());
        ids.truncate(limit);
    }
    if eos {
        ids.push(m.vocab.eos_id);
    }
    if ids.is_empty() {
        bail!(InvalidArgument, "there is no text to embed");
    }
    let n_tok = ids.len();

    let mut c = kv_init(m, n_tok, n_tok, true)?;
    model_forward(m, &mut c, &ids, 0)?;

    let width = m.embd_proj.iter().map(|p| p.rows()).fold(h.n_embd, usize::max);
    let mut a = vec![0.0f32; width];
    let mut b = vec![0.0f32; width];
    pool_rows(h.pooling, &c.s.x, n_tok, h.n_embd, &mut a);
    for p in &m.embd_proj {
        model_gemv(p, &a, &mut b);
        std::mem::swap(&mut a, &mut b);
    }
    // Matryoshka models are trained so that a prefix of the vector is itself
    // an embedding; it is normalised on its own.
    l2_normalize(&mut a[..n_out]);
    out.copy_from_slice(&a[..n_out]);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attn_spans() {
        // (causal, window, pos, n_seq) -> (first, last)
        let cases = [
            ((true, 0, 5, 10), (0, 5)), // causal: the prefix
            ((true, 3, 5, 10), (3, 5)), // sliding: the last 3 positions
            ((true, 0, 5, 5), (0, 4)),  // MTP drafter: every key the target evaluated
            ((true, 3, 5, 5), (3, 4)),
            ((false, 0, 5, 10), (0, 9)), // bidirectional: everything
            ((false, 4, 5, 10), (3, 7)), // ...within |key - pos| <= window / 2
            ((false, 4, 0, 10), (0, 2)),
            ((false, 4, 9, 10), (7, 9)),
            ((false, 512, 300, 1000), (44, 556)),
        ];
        for ((causal, window, pos, n_seq), want) in cases {
            assert_eq!(
                attn_span(causal, window, pos, n_seq),
                want,
                "{causal} {window} {pos} {n_seq}"
            );
        }
    }

    #[test]
    fn pooling_and_normalize() {
        let rows = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // three tokens of two values
        let mut out = [0.0f32; 2];
        pool_rows(Pooling::Mean, &rows, 3, 2, &mut out);
        assert_eq!(out, [3.0, 4.0]);
        pool_rows(Pooling::Cls, &rows, 3, 2, &mut out);
        assert_eq!(out, [1.0, 2.0]);
        pool_rows(Pooling::Last, &rows, 3, 2, &mut out);
        assert_eq!(out, [5.0, 6.0]);

        // A Matryoshka prefix is normalised on its own.
        let mut v = [3.0f32, 4.0, 12.0];
        l2_normalize(&mut v[..2]);
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6 && v[2] == 12.0);
        let mut z = [0.0f32; 2];
        l2_normalize(&mut z);
        assert_eq!(z, [0.0; 2]);
    }
}
