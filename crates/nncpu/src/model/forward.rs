//! The forward pass. Per batch of tokens: embed, per-layer inputs (Gemma 4),
//! the layers, the final norm; then, on request, the lm_head.

use super::kv::{KvCache, KvVec, Scratch};
use super::{ATTN_BLOCK, Arch, Layer, MAX_EXPERTS, ModelData, expert_matrix, f32s, model_embed_row, model_gemv};
use super::{attn_span, opt_weight_bytes, weight_bytes};
use crate::error::{Result, bail};
use crate::kernels::{self, attn_part};
use crate::perf::{PerfScope, PerfSlot};
use crate::pool::{self, SyncPtr};
use crate::runtime::{DType, Tensor, f32_to_bf16};

/// Batched weight gemv: `n_tok` activation vectors (x_stride apart) against
/// one pass over the weights, into `n_tok` outputs (y_stride apart).
pub(crate) fn gemv_batch(
    t: &Tensor,
    x: &[f32],
    x_stride: usize,
    y: &mut [f32],
    y_stride: usize,
    n_tok: usize,
    xsum: &mut [f32],
) {
    let (rows, cols) = (t.rows(), t.cols());
    let x = &x[..(n_tok - 1) * x_stride + cols];
    let y = &mut y[..(n_tok - 1) * y_stride + rows];
    match t.dtype {
        DType::Q8_0 => {
            let (qs, scales) = t.q8();
            kernels::gemv_q8_0_batch(qs, scales, x, x_stride, y, y_stride, rows, cols, n_tok);
        }
        DType::Q4S => {
            let nb = cols / 32;
            for tk in 0..n_tok {
                kernels::block_sums(
                    &x[tk * x_stride..tk * x_stride + cols],
                    &mut xsum[tk * nb..(tk + 1) * nb],
                );
            }
            let (qs, scales, biases) = t.q4();
            kernels::gemv_q4_batch(
                qs,
                scales,
                biases,
                x,
                x_stride,
                &xsum[..n_tok * nb],
                nb,
                y,
                y_stride,
                rows,
                cols,
                n_tok,
            );
        }
        _ => kernels::gemv_bf16_batch(t.bf16(), x, x_stride, y, y_stride, rows, cols, n_tok),
    }
}

/// The tensor of a field the loader has checked is present.
#[inline]
fn req(t: &Option<Tensor>) -> &Tensor {
    t.as_ref().expect("a tensor checked at load")
}

// ---------------------------------------------------------------------------
// Attention.
// ---------------------------------------------------------------------------

/// One layer's attention over a batch of query tokens.
struct AttnJob<'a> {
    /// [n_tok, n_head * head_dim]
    q: &'a [f32],
    /// [n_tok, n_head * head_dim]; token t's row is written by one thread.
    out: SyncPtr<f32>,
    /// The layer's cache: row pos % rows, kv_dim elements.
    k: &'a KvVec,
    v: &'a KvVec,
    rows: usize,
    /// n_chunks slots of `slot` floats.
    scores: SyncPtr<f32>,
    slot: usize,
    /// Decode: a partial per (KV head, block).
    parts: SyncPtr<f32>,
    n_tok: usize,
    pos0: usize,
    n_seq: usize,
    window: usize,
    n_head: usize,
    n_kv: usize,
    head_dim: usize,
    n_chunks: usize,
    /// Decode: the blocks of the one query's keys.
    blk0: usize,
    n_blk: usize,
    scale: f32,
    causal: bool,
}

impl AttnJob<'_> {
    /// Partial attention of query token t's heads sharing KV head `kv_h`
    /// over the part of its span [first, last] that falls in block b.
    #[allow(clippy::too_many_arguments)]
    fn attend_block(
        &self,
        t: usize,
        kv_h: usize,
        first: usize,
        last: usize,
        b: usize,
        scores: &mut [f32],
        part: &mut [f32],
    ) {
        let k0 = first.max(b * ATTN_BLOCK);
        let k1 = last.min(b * ATTN_BLOCK + ATTN_BLOCK - 1);
        let hd = self.head_dim;
        let kv_dim = self.n_kv * hd;
        let group = self.n_head / self.n_kv;
        let n_keys = k1 + 1 - k0;
        let off = (k0 % self.rows) * kv_dim + kv_h * hd;
        let span = (n_keys - 1) * kv_dim + hd;
        let q0 = t * self.n_head * hd + kv_h * group * hd;
        kernels::attn_block(
            &self.q[q0..q0 + group * hd],
            hd,
            self.k.rows(off, span),
            self.v.rows(off, span),
            kv_dim,
            n_keys,
            hd,
            group,
            self.scale,
            scores,
            part,
        );
    }

    /// Query heads that share a KV head are processed together: with GQA
    /// (Gemma 3 1B is 4 query heads to 1 KV head) that turns n_head passes
    /// over the K/V cache into one. The blocks are merged in order.
    fn attend_token(&self, t: usize, slot: &mut [f32]) {
        let (first, last) = attn_span(self.causal, self.window, self.pos0 + t, self.n_seq);
        let hd = self.head_dim;
        let group = self.n_head / self.n_kv;
        let part = group * attn_part(hd);
        let (scores, rest) = slot.split_at_mut(group * ATTN_BLOCK);
        let (acc, rest) = rest.split_at_mut(part);
        let tmp = &mut rest[..part];
        // SAFETY: token t's output row is written by this call alone.
        let out = unsafe { self.out.slice_mut(t * self.n_head * hd, self.n_head * hd) };
        let (b0, b1) = (first / ATTN_BLOCK, last / ATTN_BLOCK);
        for kv_h in 0..self.n_kv {
            self.attend_block(t, kv_h, first, last, b0, scores, acc);
            for b in b0 + 1..=b1 {
                self.attend_block(t, kv_h, first, last, b, scores, tmp);
                kernels::attn_merge(acc, tmp, hd, group);
            }
            kernels::attn_finish(&mut out[kv_h * group * hd..], hd, acc, hd, group);
        }
    }

    /// Chunk c takes tokens c, c + n_chunks, ... — interleaved, so that a
    /// causal batch's longer spans are shared out — and scores slot c.
    fn attend_chunks(&self, c0: u32, c1: u32) {
        for c in c0 as usize..c1 as usize {
            // SAFETY: slot c belongs to chunk c alone.
            let slot = unsafe { self.scores.slice_mut(c * self.slot, self.slot) };
            for t in (c..self.n_tok).step_by(self.n_chunks) {
                self.attend_token(t, slot);
            }
        }
    }

    /// Decode: one query, its (KV head, block) pairs shared out. Chunk c
    /// takes pairs c, c + n_chunks, ..., each into its own partial; the
    /// caller merges them in block order, as attend_token would.
    fn attend_blocks(&self, c0: u32, c1: u32) {
        let (first, last) = attn_span(self.causal, self.window, self.pos0, self.n_seq);
        let group = self.n_head / self.n_kv;
        let part = group * attn_part(self.head_dim);
        let n_items = self.n_kv * self.n_blk;
        for c in c0 as usize..c1 as usize {
            // SAFETY: slot c belongs to chunk c alone, partial i to item i.
            let scores = unsafe { self.scores.slice_mut(c * self.slot, group * ATTN_BLOCK) };
            for i in (c..n_items).step_by(self.n_chunks) {
                let p = unsafe { self.parts.slice_mut(i * part, part) };
                self.attend_block(0, i / self.n_blk, first, last, self.blk0 + i % self.n_blk, scores, p);
            }
        }
    }
}

/// Attention for a batch of `n_tok` tokens at positions pos0...: reads the
/// normed hidden states in s.normed, appends the batch's K/V to the layer's
/// cache, attends, and leaves the output projection in s.normed. An MTP
/// drafter's layers read `ext`, the target's cache, instead of their own.
fn attention_forward(m: &ModelData, c: &mut KvCache, ext: Option<&KvCache>, li: usize, pos0: usize, n_tok: usize) {
    let h = &m.hparams;
    let l = &m.layers[li];
    let n_embd = h.n_embd;
    let hd = l.head_dim;
    let (n_head, n_kv) = (l.n_head, l.n_head_kv);
    let q_dim = n_head * hd;
    let kv_dim = n_kv * hd;
    let freq = if l.sliding_window {
        h.rope_freq_base_swa
    } else {
        h.rope_freq_base
    };
    let reuse_kv = l.kv_layer != Some(li);
    let eps = h.rms_eps;
    let KvCache {
        s,
        bufs,
        kv_buf,
        kv_bf16,
        n_attn_slots,
        attn_slot,
        ..
    } = c;
    let s: &mut Scratch = s;
    let attn_q = req(&l.attn_q);

    {
        let mut wb = weight_bytes(attn_q);
        if !reuse_kv {
            wb += weight_bytes(req(&l.attn_k)) + opt_weight_bytes(&l.attn_v);
        }
        let _ps = PerfScope::new(PerfSlot::AttnQkv, wb);
        if l.attn_gated {
            // Each head's rows are its query, then its output gate.
            gemv_batch(attn_q, &s.normed, n_embd, &mut s.q_gate, 2 * q_dim, n_tok, &mut s.xsum);
            for r in 0..n_tok * n_head {
                s.q[r * hd..(r + 1) * hd].copy_from_slice(&s.q_gate[2 * r * hd..(2 * r + 1) * hd]);
                s.attn_gate[r * hd..(r + 1) * hd].copy_from_slice(&s.q_gate[(2 * r + 1) * hd..(2 * r + 2) * hd]);
            }
        } else {
            gemv_batch(attn_q, &s.normed, n_embd, &mut s.q, q_dim, n_tok, &mut s.xsum);
        }
        if !reuse_kv {
            gemv_batch(
                req(&l.attn_k),
                &s.normed,
                n_embd,
                &mut s.k_cur,
                kv_dim,
                n_tok,
                &mut s.xsum,
            );
            match &l.attn_v {
                Some(v) => gemv_batch(v, &s.normed, n_embd, &mut s.v_cur, kv_dim, n_tok, &mut s.xsum),
                // Gemma 4 global layers without v_proj: V is the raw K projection.
                None => s.v_cur[..kv_dim * n_tok].copy_from_slice(&s.k_cur[..kv_dim * n_tok]),
            }
        }
        // Projection biases (Qwen2).
        if l.attn_q_b.is_some() || l.attn_k_b.is_some() || l.attn_v_b.is_some() {
            for t in 0..n_tok {
                if let Some(b) = &l.attn_q_b {
                    kernels::add_inplace(&mut s.q[t * q_dim..(t + 1) * q_dim], b.f32s());
                }
                if reuse_kv {
                    continue;
                }
                if let Some(b) = &l.attn_k_b {
                    kernels::add_inplace(&mut s.k_cur[t * kv_dim..(t + 1) * kv_dim], b.f32s());
                }
                if let Some(b) = &l.attn_v_b {
                    kernels::add_inplace(&mut s.v_cur[t * kv_dim..(t + 1) * kv_dim], b.f32s());
                }
            }
        }
    }

    // Per-head Q / K norms (Gemma, Qwen3; Llama has none). Every (token,
    // head) pair is an independent length-head_dim group, so the whole batch
    // is one call.
    {
        let _ps = PerfScope::new(PerfSlot::Norm, 0);
        if let Some(qn) = &l.attn_q_norm {
            kernels::rmsnorm_gamma_multi_inplace(&mut s.q[..n_tok * q_dim], hd, qn.f32s(), eps);
        }
        if !reuse_kv {
            if let Some(kn) = &l.attn_k_norm {
                kernels::rmsnorm_gamma_multi_inplace(&mut s.k_cur[..n_tok * kv_dim], hd, kn.f32s(), eps);
            }
            // Gemma 4 also RMS-normalises V per head, with no learned gamma.
            if h.arch_id == Arch::Gemma4 {
                for g in 0..n_tok * n_kv {
                    kernels::rmsnorm_inplace(&mut s.v_cur[g * hd..(g + 1) * hd], eps);
                }
            }
        }
    }

    // RoPE on Q always; K only when freshly computed (cached K is already
    // rotated). Gemma 3 applies its linear position scaling to global layers
    // only (HF scales rope_theta, not rope_local_base_freq).
    {
        let _ps = PerfScope::new(PerfSlot::AttnRope, 0);
        let layer_rope_scale = if h.arch_id == Arch::Gemma3 && l.sliding_window {
            1.0
        } else {
            h.rope_freq_scale
        };
        let ff = l.rope_freqs.as_ref().map(|t| t.f32s());
        for t in 0..n_tok {
            let rpos = (pos0 + t) as f32 * layer_rope_scale;
            kernels::rope(
                &mut s.q[t * q_dim..(t + 1) * q_dim],
                n_head,
                hd,
                l.n_rot,
                rpos,
                freq,
                h.rope_mode,
                ff,
            );
            if !reuse_kv {
                let k = &mut s.k_cur[t * kv_dim..(t + 1) * kv_dim];
                kernels::rope(k, n_kv, hd, l.n_rot, rpos, freq, h.rope_mode, ff);
            }
        }
    }

    // Append K/V to the layer's cache (only when freshly computed). An MTP
    // drafter's layers read its target's cache instead.
    let buf_idx = kv_buf[li].expect("an attention layer reads a K/V buffer");
    if !reuse_kv {
        debug_assert!(ext.is_none());
        let buf = &mut bufs[buf_idx];
        assert!(buf.window == 0 || (l.sliding_window && buf.window == h.sliding_window));
        for t in 0..n_tok {
            let row = ((pos0 + t) % buf.rows) * kv_dim;
            let kt = &s.k_cur[t * kv_dim..(t + 1) * kv_dim];
            let vt = &s.v_cur[t * kv_dim..(t + 1) * kv_dim];
            match (&mut buf.k, &mut buf.v) {
                (KvVec::Bf16(kd), KvVec::Bf16(vd)) => {
                    for i in 0..kv_dim {
                        kd[row + i] = f32_to_bf16(kt[i]);
                        vd[row + i] = f32_to_bf16(vt[i]);
                    }
                }
                (KvVec::F32(kd), KvVec::F32(vd)) => {
                    kd[row..row + kv_dim].copy_from_slice(kt);
                    vd[row..row + kv_dim].copy_from_slice(vt);
                }
                _ => unreachable!("K and V share one element type"),
            }
        }
    }

    let (bufs, kv_bf16, n_seq) = match ext {
        Some(t) => (&t.bufs[..], t.kv_bf16, t.cur_pos),
        None => (&bufs[..], *kv_bf16, pos0 + n_tok),
    };
    let buf = &bufs[buf_idx];
    let esize: u64 = if kv_bf16 { 2 } else { 4 };
    let mut job = AttnJob {
        q: &s.q,
        out: SyncPtr::new(&mut s.attn),
        k: &buf.k,
        v: &buf.v,
        rows: buf.rows,
        scores: SyncPtr::new(&mut s.scores),
        slot: *attn_slot,
        parts: SyncPtr::new(&mut s.attn_part),
        n_tok,
        pos0,
        n_seq,
        window: if l.sliding_window { h.sliding_window } else { 0 },
        n_head,
        n_kv,
        head_dim: hd,
        n_chunks: 1,
        blk0: 0,
        n_blk: 0,
        scale: h.attention_scale,
        causal: h.causal,
    };

    let mut attn_bytes = 0u64;
    for t in 0..n_tok {
        let (first, last) = attn_span(job.causal, job.window, pos0 + t, n_seq);
        attn_bytes += 2 * (last + 1 - first) as u64 * kv_dim as u64 * esize;
    }
    {
        let _ps = PerfScope::new(PerfSlot::AttnCore, attn_bytes);
        // Short spans run here. Decode with a long span shares its (KV head,
        // block) pairs across the pool; a batch with long spans (a long
        // prompt, or an embedding model's whole sequence) shares out its
        // tokens. Each chunk has its own scores slot. Either way the blocks
        // and their merge order are the same, so the result does not depend
        // on the path taken.
        const PARALLEL_BYTES: u64 = 256 << 10;
        let n_slots = *n_attn_slots;
        if n_tok == 1 && attn_bytes >= PARALLEL_BYTES && n_slots > 1 {
            let (first, last) = attn_span(job.causal, job.window, pos0, n_seq);
            job.blk0 = first / ATTN_BLOCK;
            job.n_blk = last / ATTN_BLOCK + 1 - job.blk0;
            let n_items = n_kv * job.n_blk;
            job.n_chunks = n_items.min(n_slots);
            pool::parallel_for(job.n_chunks as u32, |c0, c1| job.attend_blocks(c0, c1));
            let group = n_head / n_kv;
            let part = group * attn_part(hd);
            let n_blk = job.n_blk;
            for kv_h in 0..n_kv {
                let parts = &mut s.attn_part[kv_h * n_blk * part..(kv_h + 1) * n_blk * part];
                let (acc, rest) = parts.split_at_mut(part);
                for b in 1..n_blk {
                    kernels::attn_merge(acc, &rest[(b - 1) * part..b * part], hd, group);
                }
                kernels::attn_finish(&mut s.attn[kv_h * group * hd..], hd, acc, hd, group);
            }
        } else {
            job.n_chunks = n_tok.min(n_slots);
            if job.n_chunks > 1 && attn_bytes >= PARALLEL_BYTES {
                pool::parallel_for(job.n_chunks as u32, |c0, c1| job.attend_chunks(c0, c1));
            } else {
                // SAFETY: slot 0, on this thread alone.
                let slot = unsafe { job.scores.slice_mut(0, job.slot) };
                for t in 0..n_tok {
                    job.attend_token(t, slot);
                }
            }
        }

        // Qwen3.5 scales each head's output by its gate.
        if l.attn_gated {
            kernels::mul_sigmoid(&mut s.attn[..n_tok * q_dim], &s.attn_gate[..n_tok * q_dim]);
        }
    }

    // Output projection: [q_dim] -> [n_embd], into s.normed (dead after the
    // QKV gemvs).
    let attn_output = req(&l.attn_output);
    let _ps = PerfScope::new(PerfSlot::AttnOut, weight_bytes(attn_output));
    gemv_batch(attn_output, &s.attn, q_dim, &mut s.normed, n_embd, n_tok, &mut s.xsum);
}

// ---------------------------------------------------------------------------
// Gated DeltaNet (Qwen 3.5, 3.6, 3.8).
// ---------------------------------------------------------------------------

/// A Gated DeltaNet layer in place of attention: reads the normed hidden
/// states in s.normed and leaves the layer's output projection there,
/// carrying the layer's recurrent state through the batch. Per token, after
/// the projections:
///   q, k, v = silu(causal_conv(x W_qkv))       q and k unit length per head
///   decay   = exp(softplus(x W_alpha + dt) * A),  beta = sigmoid(x W_beta)
///   S = decay S + beta (v - decay S k) k^T,    o = S q / sqrt(d_k)
///   out     = W_out (rmsnorm(o) * gamma * silu(x W_gate))
fn delta_net_forward(m: &ModelData, c: &mut KvCache, l: &Layer, n_tok: usize) {
    let h = &m.hparams;
    let n_embd = h.n_embd;
    let (n_k, d_k, n_v, d_v) = (h.ssm_n_k, h.ssm_d_k, h.ssm_n_v, h.ssm_d_v);
    let qk_dim = n_k * d_k;
    let v_dim = n_v * d_v;
    let conv_dim = 2 * qk_dim + v_dim;
    let hist = h.ssm_d_conv - 1;
    let state_floats = v_dim * d_k;
    let eps = h.rms_eps;
    let rs = l.rs_index.expect("a Gated DeltaNet layer has a state slot");
    let KvCache {
        s,
        conv_state,
        ssm_state,
        ..
    } = c;
    let s: &mut Scratch = s;

    // The convolution reads [its last d_conv - 1 inputs | the batch], so the
    // projection lands straight after the saved history.
    let history = &mut conv_state[rs * hist * conv_dim..(rs + 1) * hist * conv_dim];
    s.dn_conv[..hist * conv_dim].copy_from_slice(history);
    {
        let (qkv, gate) = (req(&l.ssm_qkv), req(&l.ssm_gate));
        let wb = weight_bytes(qkv) + weight_bytes(gate) + 2 * (n_v * n_embd * 4) as u64;
        let _ps = PerfScope::new(PerfSlot::DnProj, wb);
        gemv_batch(
            qkv,
            &s.normed,
            n_embd,
            &mut s.dn_conv[hist * conv_dim..],
            conv_dim,
            n_tok,
            &mut s.xsum,
        );
        gemv_batch(gate, &s.normed, n_embd, &mut s.dn_z, v_dim, n_tok, &mut s.xsum);
        let wa = f32s(&l.ssm_alpha);
        let wb = f32s(&l.ssm_beta);
        for t in 0..n_tok {
            let xt = &s.normed[t * n_embd..(t + 1) * n_embd];
            let g = &mut s.dn_gb[t * 2 * n_v..(t + 1) * 2 * n_v];
            for hv in 0..n_v {
                g[hv] = kernels::dot(&wa[hv * n_embd..(hv + 1) * n_embd], xt);
                g[n_v + hv] = kernels::dot(&wb[hv * n_embd..(hv + 1) * n_embd], xt);
            }
        }
    }

    {
        let _ps = PerfScope::new(PerfSlot::DnCore, (2 * n_tok * state_floats * 4) as u64);
        let taps = &l.ssm_conv_taps;
        kernels::conv1d_silu(
            &mut s.dn_qkv[..n_tok * conv_dim],
            &s.dn_conv[..(hist + n_tok) * conv_dim],
            taps,
            n_tok,
            conv_dim,
            hist + 1,
        );
        history.copy_from_slice(&s.dn_conv[n_tok * conv_dim..(n_tok + hist) * conv_dim]);

        let q_scale = 1.0 / (d_k as f32).sqrt();
        let dt = f32s(&l.ssm_dt);
        let a = f32s(&l.ssm_a);
        for t in 0..n_tok {
            let row = &mut s.dn_qkv[t * conv_dim..(t + 1) * conv_dim];
            kernels::l2norm_multi(&mut row[..qk_dim], d_k, eps, q_scale);
            kernels::l2norm_multi(&mut row[qk_dim..2 * qk_dim], d_k, eps, 1.0);
            let g = &mut s.dn_gb[t * 2 * n_v..(t + 1) * 2 * n_v];
            for hv in 0..n_v {
                let av = g[hv] + dt[hv];
                let softplus = if av > 20.0 { av } else { av.exp().ln_1p() };
                g[hv] = (softplus * a[hv]).exp();
                g[n_v + hv] = 1.0 / (1.0 + (-g[n_v + hv]).exp());
            }
        }

        // A value head's state depends only on its own inputs, so heads are
        // shared out across the pool, each running through the batch's
        // tokens in order.
        let state = SyncPtr::new(&mut ssm_state[rs * state_floats..(rs + 1) * state_floats]);
        let out = SyncPtr::new(&mut s.dn_out);
        let qkv = &s.dn_qkv;
        let gbs = &s.dn_gb;
        let heads = |h0: u32, h1: u32| {
            for hv in h0 as usize..h1 as usize {
                // The converter orders value heads so that head hv reads key
                // head hv mod n_k (ggml's broadcast), not hv / (n_v / n_k).
                let hk = hv % n_k;
                // SAFETY: head hv's state and outputs belong to this call alone.
                let st = unsafe { state.slice_mut(hv * d_v * d_k, d_v * d_k) };
                for t in 0..n_tok {
                    let row = &qkv[t * conv_dim..(t + 1) * conv_dim];
                    let gb = &gbs[t * 2 * n_v..(t + 1) * 2 * n_v];
                    let o = unsafe { out.slice_mut(t * v_dim + hv * d_v, d_v) };
                    kernels::delta_rule(
                        st,
                        &row[hk * d_k..(hk + 1) * d_k],
                        &row[qk_dim + hk * d_k..qk_dim + (hk + 1) * d_k],
                        &row[2 * qk_dim + hv * d_v..2 * qk_dim + (hv + 1) * d_v],
                        gb[hv],
                        gb[n_v + hv],
                        o,
                        d_k,
                        d_v,
                    );
                }
            }
        };
        // A head's state is d_v * d_k floats (64 KB at 128 x 128); below a few
        // hundred KB of work in all, dispatch would cost more than it saves.
        if n_v > 1 && n_tok * state_floats >= 64 << 10 {
            pool::parallel_for(n_v as u32, heads);
        } else {
            heads(0, n_v as u32);
        }

        let o = &mut s.dn_out[..n_tok * v_dim];
        kernels::rmsnorm_gamma_multi_inplace(o, d_v, f32s(&l.ssm_norm), eps);
        kernels::silu_mul_into(o, &s.dn_z[..n_tok * v_dim]);
    }

    let ssm_out = req(&l.ssm_out);
    let _ps = PerfScope::new(PerfSlot::DnOut, weight_bytes(ssm_out));
    gemv_batch(ssm_out, &s.dn_out, v_dim, &mut s.normed, n_embd, n_tok, &mut s.xsum);
}

// ---------------------------------------------------------------------------
// Mixture of experts (Gemma 4 26B A4B).
// ---------------------------------------------------------------------------

/// Gemma 4's routed experts for `n_tok` tokens whose residual stream after
/// attention is s.x. Leaves post_ffw_norm_2(sum over the chosen experts of
/// weight * expert(x)) in s.moe_out; the dense MLP beside it is the shared
/// expert.
fn moe_forward(m: &ModelData, s: &mut Scratch, l: &Layer, n_tok: usize) {
    let h = &m.hparams;
    let n_embd = h.n_embd;
    let n_expert = h.n_expert;
    let k = h.n_expert_used;
    let n_ff = l.n_ff_exp;
    let eps = h.rms_eps;

    // Router: softmax over rms_norm(x) / sqrt(n_embd) * scale against the
    // F32 router weights, then the k most likely experts.
    let mut used = [0u64; MAX_EXPERTS / 64];
    {
        let _ps = PerfScope::new(PerfSlot::MoeRoute, 0);
        let router = f32s(&l.ffn_gate_inp);
        let scale = f32s(&l.ffn_gate_inp_scale);
        let inv_sqrt = 1.0 / (n_embd as f32).sqrt();
        for t in 0..n_tok {
            let r = &mut s.moe_router[..n_embd];
            kernels::rmsnorm(r, &s.x[t * n_embd..(t + 1) * n_embd], eps);
            for (ri, &sc) in r.iter_mut().zip(scale) {
                *ri *= inv_sqrt * sc;
            }
            let p = &mut s.moe_probs[..n_expert];
            for (e, pe) in p.iter_mut().enumerate() {
                *pe = kernels::dot(&router[e * n_embd..(e + 1) * n_embd], r);
            }
            kernels::softmax_inplace(p);
            kernels::moe_top_k(
                p,
                k,
                &mut s.moe_expert[t * k..(t + 1) * k],
                &mut s.moe_weight[t * k..(t + 1) * k],
            );
            for &e in &s.moe_expert[t * k..(t + 1) * k] {
                used[e as usize / 64] |= 1 << (e as usize % 64);
            }
        }
    }

    let down_exps = req(&l.ffn_down_exps);
    let first = l
        .ffn_gate_up_exps
        .as_ref()
        .or(l.ffn_gate_exps.as_ref())
        .expect("checked at load");
    let mut expert_bytes = weight_bytes(first) + weight_bytes(down_exps);
    if l.ffn_gate_up_exps.is_none() {
        expert_bytes += weight_bytes(req(&l.ffn_up_exps));
    }
    let n_used: u64 = used.iter().map(|w| w.count_ones() as u64).sum();
    let _ps = PerfScope::new(PerfSlot::MoeExperts, n_used * expert_bytes);

    let n_all = n_tok * n_embd;
    kernels::rmsnorm_gamma_multi(
        &mut s.moe_in[..n_all],
        &s.x[..n_all],
        n_embd,
        f32s(&l.pre_ffw_norm_2),
        eps,
    );
    s.moe_out[..n_all].fill(0.0);

    // Each chosen expert runs once, over every token that picked it, so its
    // weights are streamed once per batch rather than once per token.
    let gu_stride = 2 * n_ff;
    for e in 0..n_expert {
        if (used[e / 64] >> (e % 64)) & 1 == 0 {
            continue;
        }
        let mut cnt = 0;
        for i in 0..n_tok * k {
            if s.moe_expert[i] == e as i32 {
                s.moe_pairs[cnt] = i;
                cnt += 1;
            }
        }
        for c in 0..cnt {
            let src = s.moe_pairs[c] / k;
            s.moe_rows[c * n_embd..(c + 1) * n_embd].copy_from_slice(&s.moe_in[src * n_embd..(src + 1) * n_embd]);
        }

        // Gate rows come first, then up rows: [gate | up] per token.
        match &l.ffn_gate_up_exps {
            Some(gu) => {
                let w = expert_matrix(gu, e);
                gemv_batch(&w, &s.moe_rows, n_embd, &mut s.moe_gate_up, gu_stride, cnt, &mut s.xsum);
            }
            None => {
                let g = expert_matrix(req(&l.ffn_gate_exps), e);
                let u = expert_matrix(req(&l.ffn_up_exps), e);
                gemv_batch(&g, &s.moe_rows, n_embd, &mut s.moe_gate_up, gu_stride, cnt, &mut s.xsum);
                gemv_batch(
                    &u,
                    &s.moe_rows,
                    n_embd,
                    &mut s.moe_gate_up[n_ff..],
                    gu_stride,
                    cnt,
                    &mut s.xsum,
                );
            }
        }
        for c in 0..cnt {
            let (gate, up) = s.moe_gate_up[c * gu_stride..(c + 1) * gu_stride].split_at_mut(n_ff);
            kernels::gelu_mul_inplace(gate, up);
        }
        let d = expert_matrix(down_exps, e);
        gemv_batch(&d, &s.moe_gate_up, gu_stride, &mut s.moe_down, n_embd, cnt, &mut s.xsum);

        let sc = l.ffn_down_exps_scale.as_ref().map_or(1.0, |t| t.f32s()[e]);
        for c in 0..cnt {
            let pair = s.moe_pairs[c];
            let w = s.moe_weight[pair] * sc;
            let o = &mut s.moe_out[(pair / k) * n_embd..(pair / k + 1) * n_embd];
            for (oi, &di) in o.iter_mut().zip(&s.moe_down[c * n_embd..(c + 1) * n_embd]) {
                *oi += w * di;
            }
        }
    }
    kernels::rmsnorm_gamma_multi_inplace(&mut s.moe_out[..n_all], n_embd, f32s(&l.post_ffw_norm_2), eps);
}

// ---------------------------------------------------------------------------
// Layers and the model.
// ---------------------------------------------------------------------------

/// Runs `n_tok` consecutive tokens (positions pos0 .. pos0 + n_tok - 1)
/// through one transformer block. The residual stream s.x is [n_tok, n_embd],
/// token-major, updated in place; n_tok == 1 is the decode path. `ple` is the
/// layer's slice of s.ple_inputs: (offset, stride between tokens).
pub(super) fn layer_forward(
    m: &ModelData,
    c: &mut KvCache,
    ext: Option<&KvCache>,
    li: usize,
    pos0: usize,
    n_tok: usize,
    ple: Option<(usize, usize)>,
) {
    let h = &m.hparams;
    let l = &m.layers[li];
    let n_embd = h.n_embd;
    let eps = h.rms_eps;
    let n_all = n_tok * n_embd;

    // --- attention block, or a Gated DeltaNet in its place ---
    {
        let _ps = PerfScope::new(PerfSlot::Norm, 0);
        let s = &mut c.s;
        kernels::rmsnorm_gamma_multi(&mut s.normed[..n_all], &s.x[..n_all], n_embd, f32s(&l.attn_norm), eps);
    }
    if l.rs_index.is_some() {
        delta_net_forward(m, c, l, n_tok);
    } else {
        attention_forward(m, c, ext, li, pos0, n_tok);
    }
    let s = &mut c.s;

    // Post-attention RMSNorm (Gemma) + residual. The attention output is in
    // s.normed.
    {
        let _ps = PerfScope::new(PerfSlot::Norm, 0);
        if let Some(pn) = &l.post_attention_norm {
            kernels::rmsnorm_gamma_multi_inplace(&mut s.normed[..n_all], n_embd, pn.f32s(), eps);
        }
        kernels::add_inplace(&mut s.x[..n_all], &s.normed[..n_all]);
    }

    // --- MLP block (gated) ---
    // Gemma uses gelu(gate)*up; Llama and Qwen use silu(gate)*up (SwiGLU).
    {
        let _ps = PerfScope::new(PerfSlot::Norm, 0);
        kernels::rmsnorm_gamma_multi(&mut s.normed[..n_all], &s.x[..n_all], n_embd, f32s(&l.ffn_norm), eps);
    }
    let n_ff = l.n_ff;
    let (ffn_gate, ffn_up, ffn_down) = (req(&l.ffn_gate), req(&l.ffn_up), req(&l.ffn_down));
    {
        let _ps = PerfScope::new(PerfSlot::FfnGateUp, weight_bytes(ffn_gate) + weight_bytes(ffn_up));
        gemv_batch(ffn_gate, &s.normed, n_embd, &mut s.gate, n_ff, n_tok, &mut s.xsum);
        gemv_batch(ffn_up, &s.normed, n_embd, &mut s.up, n_ff, n_tok, &mut s.xsum);
    }
    {
        let _ps = PerfScope::new(PerfSlot::FfnAct, 0);
        let n_ff_all = n_tok * n_ff;
        if matches!(h.arch_id, Arch::Gemma3 | Arch::Gemma4) {
            kernels::gelu_mul_inplace(&mut s.gate[..n_ff_all], &s.up[..n_ff_all]);
        } else {
            kernels::swiglu_inplace(&mut s.gate[..n_ff_all], &s.up[..n_ff_all]);
        }
    }
    {
        let _ps = PerfScope::new(PerfSlot::FfnDown, weight_bytes(ffn_down));
        gemv_batch(ffn_down, &s.gate, n_ff, &mut s.normed, n_embd, n_tok, &mut s.xsum);
    }
    // The MLP output is in s.normed.

    // Gemma 4 MoE: the dense MLP above is the shared expert. Both halves are
    // normalised on their own, then summed.
    if l.n_ff_exp > 0 {
        {
            let _ps = PerfScope::new(PerfSlot::Norm, 0);
            kernels::rmsnorm_gamma_multi_inplace(&mut s.normed[..n_all], n_embd, f32s(&l.post_ffw_norm_1), eps);
        }
        moe_forward(m, s, l, n_tok);
        kernels::add_inplace(&mut s.normed[..n_all], &s.moe_out[..n_all]);
    }

    // Post-FFW RMSNorm (Gemma) + residual.
    {
        let _ps = PerfScope::new(PerfSlot::Norm, 0);
        if let Some(pf) = &l.post_ffw_norm {
            kernels::rmsnorm_gamma_multi_inplace(&mut s.normed[..n_all], n_embd, pf.f32s(), eps);
        }
        kernels::add_inplace(&mut s.x[..n_all], &s.normed[..n_all]);
    }

    // --- Per-Layer Embedding (PLE) block (Gemma 4) ---
    //   g = gelu(inp_gate(x)) * inp_per_layer[li];  x += post_norm(proj(g))
    if let (Some(inp_gate), Some(proj), Some((off, stride))) = (&l.inp_gate, &l.proj, ple) {
        let _ps = PerfScope::new(PerfSlot::PleLayer, weight_bytes(inp_gate) + weight_bytes(proj));
        let ple_dim = inp_gate.ne[1];
        gemv_batch(inp_gate, &s.x, n_embd, &mut s.ple_gate, ple_dim, n_tok, &mut s.xsum);
        kernels::gelu_inplace(&mut s.ple_gate[..n_tok * ple_dim]);
        for t in 0..n_tok {
            let p0 = off + t * stride;
            kernels::mul_inplace(
                &mut s.ple_gate[t * ple_dim..(t + 1) * ple_dim],
                &s.ple_inputs[p0..p0 + ple_dim],
            );
        }
        gemv_batch(
            proj,
            &s.ple_gate,
            ple_dim,
            &mut s.ple_contrib,
            n_embd,
            n_tok,
            &mut s.xsum,
        );
        if let Some(pn) = &l.post_norm {
            kernels::rmsnorm_gamma_multi_inplace(&mut s.ple_contrib[..n_all], n_embd, pn.f32s(), eps);
        }
        kernels::add_inplace(&mut s.x[..n_all], &s.ple_contrib[..n_all]);
    }

    // --- per-layer output scale (Gemma 4 layer_scalar) ---
    if let Some(scale) = &l.layer_output_scale {
        let v = scale.f32s()[0];
        for x in &mut s.x[..n_all] {
            *x *= v;
        }
    }
}

/// Run `tokens` (at most c.n_batch) at positions [pos0, pos0 + n), pos0 ==
/// c.cur_pos, appending their K/V and advancing c.cur_pos. Afterwards
/// [`model_hidden`] gives each token's final hidden state, the input to the
/// lm_head. A bidirectional model takes its whole sequence in one call, from
/// position 0.
pub(crate) fn model_forward(m: &ModelData, c: &mut KvCache, tokens: &[i32], pos0: usize) -> Result<()> {
    let h = &m.hparams;
    let n_tok = tokens.len();
    if n_tok == 0 || n_tok > c.n_batch {
        bail!(
            InvalidArgument,
            "{n_tok} tokens in one forward pass (at most {})",
            c.n_batch
        );
    }
    if pos0 != c.cur_pos {
        bail!(
            InvalidArgument,
            "position {pos0} does not follow the cache ({})",
            c.cur_pos
        );
    }
    if !h.causal && pos0 != 0 {
        bail!(
            InvalidArgument,
            "a bidirectional model takes its whole sequence in one pass"
        );
    }
    if pos0 + n_tok > c.n_ctx {
        bail!(
            ContextFull,
            "{n_tok} more tokens do not fit the {}-token context",
            c.n_ctx
        );
    }
    for &t in tokens {
        if t < 0 || t as usize >= h.n_vocab {
            bail!(
                InvalidArgument,
                "token {t} is outside the vocabulary [0, {})",
                h.n_vocab
            );
        }
    }
    // A sequence that starts again starts its recurrent state afresh.
    if pos0 == 0 {
        c.conv_state.fill(0.0);
        c.ssm_state.fill(0.0);
    }

    let n_embd = h.n_embd;
    let eps = h.rms_eps;
    let ple_dim = h.ple_dim;
    let ple_total = ple_dim * h.n_layer;
    {
        let _ps = PerfScope::new(PerfSlot::Embed, 0);
        for (t, &tok) in tokens.iter().enumerate() {
            model_embed_row(
                m.token_embd(),
                &mut c.s.x[t * n_embd..(t + 1) * n_embd],
                tok as usize,
                h.embd_scale,
            );
        }
    }

    if ple_dim > 0 {
        // Per-layer inputs: rms_norm(per_layer_model_proj(x) / sqrt(n_embd)),
        // averaged with the token's own row of per_layer_token_embd (scaled
        // by sqrt(ple_dim)) when the model has that table.
        let _ps = PerfScope::new(PerfSlot::PlePrep, opt_weight_bytes(&m.per_layer_model_proj));
        let s = &mut c.s;
        let proj = req(&m.per_layer_model_proj);
        let gamma = f32s(&m.per_layer_proj_norm);
        let ple_all = n_tok * ple_total;
        let inv_sqrt_n_embd = 1.0 / (n_embd as f32).sqrt();
        // Each layer's slice is an independent RMSNorm group, and the gamma
        // is shared, so the whole batch is one call.
        let finish = |bproj: &mut [f32]| {
            for v in bproj.iter_mut() {
                *v *= inv_sqrt_n_embd;
            }
            kernels::rmsnorm_gamma_multi_inplace(bproj, ple_dim, gamma, eps);
        };
        match &m.per_layer_token_embd {
            Some(table) => {
                let tok_scale = (ple_dim as f32).sqrt();
                for (t, &tok) in tokens.iter().enumerate() {
                    let row = &mut s.ple_inputs[t * ple_total..(t + 1) * ple_total];
                    model_embed_row(table, row, tok as usize, tok_scale);
                }
                gemv_batch(proj, &s.x, n_embd, &mut s.ple_proj, ple_total, n_tok, &mut s.xsum);
                finish(&mut s.ple_proj[..ple_all]);
                let inv_sqrt2 = 1.0 / 2.0f32.sqrt();
                for (pi, &bp) in s.ple_inputs[..ple_all].iter_mut().zip(&s.ple_proj[..ple_all]) {
                    *pi = (bp + *pi) * inv_sqrt2;
                }
            }
            None => {
                gemv_batch(proj, &s.x, n_embd, &mut s.ple_inputs, ple_total, n_tok, &mut s.xsum);
                finish(&mut s.ple_inputs[..ple_all]);
            }
        }
    }

    for li in 0..h.n_layer {
        let ple = (ple_dim > 0).then_some((li * ple_dim, ple_total));
        layer_forward(m, c, None, li, pos0, n_tok, ple);
    }

    if let Some(on) = &m.output_norm {
        let _ps = PerfScope::new(PerfSlot::Norm, 0);
        kernels::rmsnorm_gamma_multi_inplace(&mut c.s.x[..n_tok * n_embd], n_embd, on.f32s(), eps);
    }

    c.cur_pos = pos0 + n_tok;
    c.kv_written = if pos0 == 0 {
        c.cur_pos
    } else {
        c.kv_written.max(c.cur_pos)
    };
    Ok(())
}

/// The final hidden state of token `i` of the last model_forward() batch.
pub(crate) fn model_hidden<'c>(m: &ModelData, c: &'c KvCache, i: usize) -> &'c [f32] {
    let n = m.hparams.n_embd;
    &c.s.x[i * n..(i + 1) * n]
}

/// lm_head: logits (n_vocab floats, soft-capped) for a final hidden state.
pub(crate) fn model_logits(m: &ModelData, hidden: &[f32], logits: &mut [f32]) {
    let lm = m.lm_head();
    let n_vocab = m.hparams.n_vocab;
    {
        let _ps = PerfScope::new(PerfSlot::LmHead, weight_bytes(lm));
        model_gemv(lm, hidden, &mut logits[..n_vocab]);
    }
    let cap = m.hparams.final_logit_softcap;
    if cap > 0.0 {
        kernels::softcap_inplace(&mut logits[..n_vocab], cap);
    }
}

/// argmax of the logits without soft-capping them (it is monotonic).
/// `scratch` must hold n_vocab floats; the BF16 path streams instead.
pub(crate) fn model_argmax(m: &ModelData, hidden: &[f32], scratch: &mut [f32]) -> usize {
    let lm = m.lm_head();
    let _ps = PerfScope::new(PerfSlot::LmHead, weight_bytes(lm));
    let (rows, cols) = (lm.rows(), lm.cols());
    // BF16 streams the projection through a running argmax; the quantized
    // formats write the logits (a ~1 MB pass, small next to the gemv).
    if lm.dtype == DType::BF16 {
        return kernels::gemv_bf16_argmax(lm.bf16(), &hidden[..cols], rows, cols);
    }
    model_gemv(lm, hidden, &mut scratch[..rows]);
    argmax(&scratch[..rows])
}

/// The index of the first largest value.
pub(crate) fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best
}

/// lm_head for the first `n` final hidden states the last model_forward()
/// left in `c`, in one pass over the weights: n rows of n_vocab logits,
/// soft-capped if `soft_cap`. Each row is what model_logits() gives for that
/// hidden state alone (Q8_0 and Q4 weights).
pub(crate) fn model_logits_batch(m: &ModelData, c: &mut KvCache, n: usize, logits: &mut [f32], soft_cap: bool) {
    let lm = m.lm_head();
    let n_vocab = m.hparams.n_vocab;
    {
        let _ps = PerfScope::new(PerfSlot::LmHead, weight_bytes(lm));
        gemv_batch(lm, &c.s.x, m.hparams.n_embd, logits, n_vocab, n, &mut c.s.xsum);
    }
    let cap = m.hparams.final_logit_softcap;
    if soft_cap && cap > 0.0 {
        for row in logits[..n * n_vocab].chunks_exact_mut(n_vocab) {
            kernels::softcap_inplace(row, cap);
        }
    }
}
