//! Views of a loaded model's internals for the command-line app's diagnostic
//! modes (`nncpu --gemma-info`, `--gemma-probe` and `--inspect-model`).
//!
//! Hidden from the documentation and not part of the supported API: what
//! these functions report, and the functions themselves, may change in any
//! release.

use crate::error::{Result, bail};
use crate::kernels::{self, RopeMode};
use crate::model::{Layer, ModelData, model_embed_row, model_gemv};
use crate::runtime::Tensor;
use crate::tokenizer::ChatFormat;
use crate::{Model, Token};
use std::fmt::Write;

/// Whether nncpu can load tensors of the ggml type `ggml_type`: F32, F16,
/// BF16, and the block quantizations its loader decodes.
pub fn tensor_type_supported(ggml_type: u32) -> bool {
    matches!(ggml_type, 0 | 1 | 30) || kernels::quant_block_elems(ggml_type) > 0
}

/// What `nncpu --gemma-info` prints: the hyperparameters, the vocabulary and
/// chat format, and which weights the layers have.
pub fn model_report(model: &Model) -> String {
    let f = &*model.0;
    let h = &f.hparams;
    let v = &f.vocab;
    // Writing to a String cannot fail, so the results of write! are ignored.
    let mut o = String::new();
    let _ = writeln!(
        o,
        "{}: arch={} file_type={} weights={}",
        f.name,
        h.arch,
        h.file_type,
        f.weights.name()
    );
    let _ = writeln!(
        o,
        "  n_layer={} n_embd={} n_head={} n_head_kv={} head_dim={} (swa={})",
        h.n_layer, h.n_embd, h.n_head, h.n_head_kv, h.head_dim, h.head_dim_swa
    );
    let _ = writeln!(
        o,
        "  rope: dim={} (swa={}) freq_base={} (swa={}) scale={} mode={} freq_factors={}",
        h.rope_dim,
        h.rope_dim_swa,
        g(h.rope_freq_base),
        g(h.rope_freq_base_swa),
        g(h.rope_freq_scale),
        if h.rope_mode == RopeMode::Norm {
            "normal"
        } else {
            "neox"
        },
        if f.rope_freqs.is_some() { "yes" } else { "no" }
    );
    let _ = writeln!(
        o,
        "  ctx={} sliding_window={} shared_kv_layers={} ple_dim={} softcap={} rms_eps={}",
        h.n_ctx,
        h.sliding_window,
        h.shared_kv_layers,
        h.ple_dim,
        g(h.final_logit_softcap),
        g(h.rms_eps)
    );
    let _ = writeln!(
        o,
        "  attention_scale={} embd_scale={}",
        g(h.attention_scale),
        g(h.embd_scale)
    );
    if h.n_layer_rs > 0 {
        let _ = writeln!(
            o,
            "  gated deltanet: {} layers, conv={}, {} q/k heads of {}, {} v heads of {}",
            h.n_layer_rs, h.ssm_d_conv, h.ssm_n_k, h.ssm_d_k, h.ssm_n_v, h.ssm_d_v
        );
    }

    let chat = match v.chat {
        ChatFormat::Gemma => "gemma",
        ChatFormat::Gemma4 => "gemma4",
        ChatFormat::Llama3 => "llama3",
        ChatFormat::ChatMl => "chatml",
        ChatFormat::None => "none",
    };
    let _ = write!(
        o,
        "  vocab={} tokens, {} (bos={} eos={} unk={} pad={}), chat={chat}, end-of-generation:",
        h.n_vocab, v.model, v.bos_id, v.eos_id, v.unk_id, v.pad_id
    );
    for &id in &v.eog_ids {
        let _ = write!(o, " {id}'{}'", token_text(f, id));
    }
    if v.can_think || v.always_thinks {
        let open = if v.always_thinks {
            &v.reply_open
        } else {
            &v.reply_open_think
        }
        .replace('\n', "\\n");
        let system = if v.tok_think >= 0 {
            "<|think|>".to_string()
        } else if v.think_instructions.is_empty() {
            "-".to_string()
        } else {
            format!("\"{}\"", v.think_instructions)
        };
        let when = if v.always_thinks { "always" } else { "with --think" };
        let _ = write!(
            o,
            "\n  thinking: {when}; the reply opens \"{open}\"; the system turn starts {system}"
        );
    }
    let _ = writeln!(
        o,
        "\n  {:.1} MB decoded, {} parameters",
        f.decoded_bytes as f64 / (1024.0 * 1024.0),
        f.n_params
    );

    let _ = writeln!(o, "\nper-layer tensor presence:");
    type Weight = fn(&Layer) -> &Option<Tensor>;
    let weights: [(&str, Weight); 20] = [
        ("attn_norm", |l| &l.attn_norm),
        ("ffn_norm", |l| &l.ffn_norm),
        ("post_attention_norm", |l| &l.post_attention_norm),
        ("post_ffw_norm", |l| &l.post_ffw_norm),
        ("post_norm", |l| &l.post_norm),
        ("attn_q_norm", |l| &l.attn_q_norm),
        ("attn_k_norm", |l| &l.attn_k_norm),
        ("attn_q", |l| &l.attn_q),
        ("attn_k", |l| &l.attn_k),
        ("attn_v", |l| &l.attn_v),
        ("attn_output", |l| &l.attn_output),
        ("attn_q.bias", |l| &l.attn_q_b),
        ("attn_qkv (deltanet)", |l| &l.ssm_qkv),
        ("ssm_out (deltanet)", |l| &l.ssm_out),
        ("ffn_gate", |l| &l.ffn_gate),
        ("ffn_up", |l| &l.ffn_up),
        ("ffn_down", |l| &l.ffn_down),
        ("inp_gate", |l| &l.inp_gate),
        ("proj", |l| &l.proj),
        ("layer_output_scale", |l| &l.layer_output_scale),
    ];
    for (name, weight) in weights {
        let n = f.layers.iter().filter(|l| weight(l).is_some()).count();
        let tag = if n == h.n_layer {
            "ok"
        } else if n == 0 {
            "absent"
        } else {
            "partial"
        };
        let _ = writeln!(o, "  layer.{name:<22} : {n:>3} / {}  [{tag}]", h.n_layer);
    }
    if h.n_layer_kv < h.n_layer {
        let _ = writeln!(
            o,
            "  (layers {}..{} reuse earlier layers' K/V, so load no K/V weights)",
            h.n_layer_kv,
            h.n_layer - 1
        );
    }

    let present = |t: &Option<Tensor>, missing| if t.is_some() { "ok" } else { missing };
    let _ = writeln!(
        o,
        "\ntop-level: token_embd={} output={} output_norm={} rope_freqs={} per_layer_token_embd={}",
        present(&f.token_embd, "absent"),
        present(&f.output, "tied"),
        present(&f.output_norm, "absent"),
        present(&f.rope_freqs, "absent"),
        present(&f.per_layer_token_embd, "absent")
    );

    let (min_ff, max_ff) = span(f.layers.iter().map(|l| l.n_ff));
    // A Gated DeltaNet layer has no attention heads.
    let (min_hd, max_hd) = span(f.layers.iter().filter(|l| l.rs_index.is_none()).map(|l| l.head_dim));
    let sliding = f.layers.iter().filter(|l| l.sliding_window).count();
    let _ = writeln!(o, "sliding-window layers: {sliding} / {}", h.n_layer);
    let _ = writeln!(o, "per-layer n_ff: {min_ff}..{max_ff}, head_dim: {min_hd}..{max_hd}");
    o
}

/// What `nncpu --gemma-probe` prints: the first attention layer worked by hand
/// for `token` (its embedding row, attn_norm, the Q projection, the Q norm and
/// RoPE at position 0), with statistics after each step.
pub fn probe(model: &Model, token: Token) -> Result<String> {
    let f = &*model.0;
    let h = &f.hparams;
    let mut o = String::new();
    let Some(li) = f.layers.iter().position(|l| l.rs_index.is_none()) else {
        o.push_str("probe: the model has no attention layers\n");
        return Ok(o);
    };
    let l = &f.layers[li];
    let (Some(token_embd), Some(attn_norm), Some(attn_q)) = (&f.token_embd, &l.attn_norm, &l.attn_q) else {
        bail!(UnsupportedModel, "layer {li} lacks the weights the probe runs");
    };
    let Some(label) = usize::try_from(token).ok().and_then(|t| f.vocab.tokens.get(t)) else {
        bail!(
            InvalidArgument,
            "token {token} is outside the vocabulary [0, {})",
            f.vocab.tokens.len()
        );
    };
    let q_dim = l.n_head * l.head_dim;
    let _ = writeln!(o, "probe: token {token} ('{label}')");
    let _ = writeln!(
        o,
        "  layer {li}: sliding_window={} q_dim={q_dim} head_dim={} n_head={} n_rot={} gated={}",
        u8::from(l.sliding_window),
        l.head_dim,
        l.n_head,
        l.n_rot,
        u8::from(l.attn_gated)
    );

    let mut x = vec![0.0; h.n_embd];
    let mut normed = vec![0.0; h.n_embd];
    let mut q = vec![0.0; attn_q.rows()];
    model_embed_row(token_embd, &mut x, token as usize, h.embd_scale);
    stats(&mut o, "embed * scale", &x);
    kernels::rmsnorm_gamma_multi(&mut normed, &x, h.n_embd, attn_norm.f32s(), h.rms_eps);
    stats(&mut o, "rmsnorm * attn_norm", &normed);
    model_gemv(attn_q, &normed, &mut q);
    if let Some(bias) = &l.attn_q_b {
        kernels::add_inplace(&mut q, bias.f32s());
    }
    stats(&mut o, "Q = W_q * normed", &q);
    if l.attn_gated {
        // Each head's rows are its query, then its output gate: keep the queries.
        let hd = l.head_dim;
        for head in 0..l.n_head {
            q.copy_within(2 * head * hd..(2 * head + 1) * hd, head * hd);
        }
        q.truncate(q_dim);
    }
    if let Some(q_norm) = &l.attn_q_norm {
        kernels::rmsnorm_gamma_multi_inplace(&mut q, l.head_dim, q_norm.f32s(), h.rms_eps);
        stats(&mut o, "Q after q_norm", &q);
    }
    let freq_base = if l.sliding_window {
        h.rope_freq_base_swa
    } else {
        h.rope_freq_base
    };
    let freq_factors = l.rope_freqs.as_ref().map(Tensor::f32s);
    kernels::rope(
        &mut q,
        l.n_head,
        l.head_dim,
        l.n_rot,
        0.0,
        freq_base,
        h.rope_mode,
        freq_factors,
    );
    stats(&mut o, "Q after RoPE (pos 0)", &q);
    Ok(o)
}

/// The smallest and largest of `sizes`; INT_MAX..0 when there are none.
fn span(sizes: impl Iterator<Item = usize>) -> (usize, usize) {
    sizes.fold((i32::MAX as usize, 0), |(lo, hi), n| (lo.min(n), hi.max(n)))
}

fn token_text(f: &ModelData, id: Token) -> &str {
    usize::try_from(id)
        .ok()
        .and_then(|i| f.vocab.tokens.get(i))
        .map_or("?", String::as_str)
}

/// One line of value statistics: range, mean and root mean square.
fn stats(o: &mut String, tag: &str, values: &[f32]) {
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    let (mut sum, mut sum2) = (0.0f64, 0.0f64);
    for &v in values {
        if v < lo {
            lo = v;
        }
        if hi < v {
            hi = v;
        }
        sum += f64::from(v);
        sum2 += f64::from(v) * f64::from(v);
    }
    let n = values.len();
    let mean = if n > 0 { sum / n as f64 } else { 0.0 };
    let rms = if n > 0 { (sum2 / n as f64).sqrt() } else { 0.0 };
    let signed = |v: f64| {
        let s = fmt_g(v, 4);
        if s.starts_with('-') { s } else { format!("+{s}") }
    };
    let _ = writeln!(
        o,
        "  {tag:<26} n={n:<6} min={} max={} mean={} rms={}",
        signed(f64::from(lo)),
        signed(f64::from(hi)),
        signed(mean),
        fmt_g(rms, 4)
    );
}

fn g(v: f32) -> String {
    fmt_g(f64::from(v), 6)
}

/// `v` as C's printf("%.*g", precision, v) writes it: the shorter of fixed and
/// exponent notation at `precision` significant digits, trailing zeros removed.
fn fmt_g(v: f64, precision: usize) -> String {
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
    let sci = format!("{:.*e}", p - 1, v);
    let (mantissa, exp) = sci.split_once('e').expect("exponent notation has an 'e'");
    let exp: i32 = exp.parse().expect("the exponent is an integer");
    let trim = |s: &str| {
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            s.to_string()
        }
    };
    if exp < -4 || exp >= p as i32 {
        format!(
            "{}e{}{:02}",
            trim(mantissa),
            if exp < 0 { '-' } else { '+' },
            exp.unsigned_abs()
        )
    } else {
        trim(&format!("{:.*}", (p as i32 - 1 - exp) as usize, v))
    }
}

#[cfg(test)]
mod tests {
    use super::fmt_g;

    #[test]
    fn printf_g() {
        assert_eq!(fmt_g(1e-6, 6), "1e-06");
        assert_eq!(fmt_g(1e6, 6), "1e+06");
        assert_eq!(fmt_g(10000.0, 6), "10000");
        assert_eq!(fmt_g(0.0001, 6), "0.0001");
        assert_eq!(fmt_g(0.0625, 6), "0.0625");
        assert_eq!(fmt_g(25.6, 6), "25.6");
        assert_eq!(fmt_g(-0.001369, 4), "-0.001369");
        assert_eq!(fmt_g(0.0, 6), "0");
    }
}
