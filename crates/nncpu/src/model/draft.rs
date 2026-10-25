//! MTP drafting with a Gemma 4 "assistant" drafter: four small layers that
//! predict the next few tokens from the target's own state, attending to the
//! target's K/V cache.

use super::forward::layer_forward;
use super::kv::{KvCache, kv_init};
use super::{Arch, ModelData, f32s, model_embed_row, model_gemv, weight_bytes};
use crate::error::{Result, bail};
use crate::kernels;
use crate::perf::{PerfSlot, perf_add, perf_enabled, perf_set};
use std::time::Instant;

/// A cache for `d` drafting for `t`, whose cache is `tcache`: the drafter's
/// sliding layers read the cache of the target's last sliding layer, and its
/// global layers that of the last global layer. Fails, with the reason, when
/// the drafter was not made for the target.
pub(crate) fn draft_init(d: &ModelData, t: &ModelData, tcache: &KvCache) -> Result<KvCache> {
    let (dh, th) = (&d.hparams, &t.hparams);
    if dh.n_embd_target == 0 {
        bail!(InvalidArgument, "{} is not an MTP drafter", d.name);
    }
    if th.arch_id != Arch::Gemma4
        || !th.causal
        || th.n_embd_target > 0
        || dh.n_embd_target != th.n_embd
        || dh.n_vocab != th.n_vocab
    {
        bail!(
            UnsupportedModel,
            "the drafter {} was not made for {} (it drafts for a Gemma 4 model {} wide)",
            d.name,
            t.name,
            dh.n_embd_target
        );
    }
    let (mut last_swa, mut last_global) = (None, None);
    for (li, l) in t.layers.iter().enumerate() {
        if l.sliding_window {
            last_swa = Some(li);
        } else {
            last_global = Some(li);
        }
    }
    let mut dc = kv_init(d, tcache.n_ctx, 1, false)?;
    for (li, l) in d.layers.iter().enumerate() {
        let tl = if l.sliding_window { last_swa } else { last_global };
        let matches = tl.is_some_and(|tl| {
            l.head_dim == t.layers[tl].head_dim
                && l.n_head_kv == t.layers[tl].n_head_kv
                && (!l.sliding_window || dh.sliding_window == th.sliding_window)
        });
        if !matches {
            bail!(
                UnsupportedModel,
                "the drafter {} does not match the attention of {}",
                d.name,
                t.name
            );
        }
        dc.kv_buf[li] = tcache.kv_buf[tl.unwrap()];
    }
    Ok(dc)
}

/// One draft step at position `pos`: `token` is the next token, which the
/// target has not evaluated yet (its cache holds positions [0, pos)), and `h`
/// the target's final hidden state before it, or the previous step's output.
/// Returns the drafter's guess for the token after `token` and the
/// probability it gives that guess, and replaces `h` (n_embd_target floats)
/// with the next step's input. Every step of one round uses the same `pos`.
pub(crate) fn draft_step(
    d: &ModelData,
    t: &ModelData,
    dc: &mut KvCache,
    tcache: &KvCache,
    token: i32,
    h: &mut [f32],
    pos: usize,
) -> (i32, f32) {
    let dh = &d.hparams;
    let n_tgt = dh.n_embd_target;
    let n_embd = dh.n_embd;
    let lm = d.lm_head();
    let nextn_pre = d.nextn_pre.as_ref().expect("a drafter has nextn.pre_projection");
    let nextn_post = d.nextn_post.as_ref().expect("a drafter has nextn.post_projection");

    // The step is profiled as one phase: its layers' own scopes are off.
    let perf = perf_enabled();
    let t0 = perf.then(Instant::now);
    perf_set(false);

    // [the target's embedding of the token | the hidden state before it]
    {
        let s = &mut dc.s;
        model_embed_row(
            t.token_embd(),
            &mut s.draft_in[..n_tgt],
            token as usize,
            t.hparams.embd_scale,
        );
        s.draft_in[n_tgt..2 * n_tgt].copy_from_slice(&h[..n_tgt]);
        model_gemv(nextn_pre, &s.draft_in, &mut s.x[..n_embd]);
    }

    // All steps of a round share the position of the first token.
    for li in 0..dh.n_layer {
        layer_forward(d, dc, Some(tcache), li, pos, 1, None);
    }
    let s = &mut dc.s;
    kernels::rmsnorm_gamma_multi_inplace(&mut s.x[..n_embd], n_embd, f32s(&d.output_norm), dh.rms_eps);
    model_gemv(nextn_post, &s.x[..n_embd], &mut h[..n_tgt]);

    let n_vocab = dh.n_vocab;
    let logits = &mut s.draft_logits[..n_vocab];
    model_gemv(lm, &s.x[..n_embd], logits);
    let best = super::forward::argmax(logits);
    kernels::softmax_inplace(logits);
    let p = logits[best];

    perf_set(perf);
    if let Some(t0) = t0 {
        let mut bytes = weight_bytes(nextn_pre) + weight_bytes(nextn_post) + weight_bytes(lm);
        for l in &d.layers {
            for w in [&l.attn_q, &l.attn_output, &l.ffn_gate, &l.ffn_up, &l.ffn_down]
                .into_iter()
                .flatten()
            {
                bytes += weight_bytes(w);
            }
        }
        perf_add(PerfSlot::Draft, t0.elapsed().as_nanos() as u64, bytes);
    }
    (best as i32, p)
}
