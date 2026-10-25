//! The phase profiler.
//!
//! A fixed-slot wall-clock accumulator, off by default and free when off.
//! When on, every kernel phase adds its wall time and the bytes it streamed,
//! from which [`perf_report`] derives an effective bandwidth. Process-wide,
//! so it is meant for one session at a time.

use std::fmt::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PerfSlot {
    /// Token embedding row lookup.
    Embed,
    /// Per-layer-input embedding table + projection.
    PlePrep,
    /// RMSNorm + gamma (all sites).
    Norm,
    /// Q/K/V projections.
    AttnQkv,
    /// Per-head Q/K norms + RoPE.
    AttnRope,
    /// Q.K scores + softmax * V.
    AttnCore,
    /// Attention output projection.
    AttnOut,
    /// Gated DeltaNet input projections.
    DnProj,
    /// Gated DeltaNet convolution, recurrence and gated norm.
    DnCore,
    /// Gated DeltaNet output projection.
    DnOut,
    /// Gate + up projections.
    FfnGateUp,
    /// The gated activation.
    FfnAct,
    /// Down projection.
    FfnDown,
    /// Mixture-of-experts router + top-k.
    MoeRoute,
    /// The chosen experts' gated MLPs.
    MoeExperts,
    /// Per-layer input gate + projection.
    PleLayer,
    /// Final logits projection (+ argmax).
    LmHead,
    /// MTP draft steps, whole.
    Draft,
}

const N_SLOTS: usize = PerfSlot::Draft as usize + 1;

const NAMES: [&str; N_SLOTS] = [
    "embed",
    "ple_prep",
    "norm",
    "attn_qkv",
    "attn_rope",
    "attn_core",
    "attn_out",
    "dn_proj",
    "dn_core",
    "dn_out",
    "ffn_gate_up",
    "ffn_act",
    "ffn_down",
    "moe_route",
    "moe_experts",
    "ple_layer",
    "lm_head",
    "mtp_draft",
];

struct SlotData {
    ns: AtomicU64,
    bytes: AtomicU64,
    calls: AtomicU64,
}

#[allow(clippy::declare_interior_mutable_const)]
const EMPTY: SlotData = SlotData {
    ns: AtomicU64::new(0),
    bytes: AtomicU64::new(0),
    calls: AtomicU64::new(0),
};

static PERF_ON: AtomicBool = AtomicBool::new(false);
static SLOTS: [SlotData; N_SLOTS] = [EMPTY; N_SLOTS];

#[inline(always)]
pub(crate) fn perf_enabled() -> bool {
    PERF_ON.load(Relaxed)
}

/// Pause or resume collection without touching the totals (an MTP draft
/// step is timed whole, so its layers' own scopes are switched off).
pub(crate) fn perf_set(on: bool) {
    PERF_ON.store(on, Relaxed);
}

pub(crate) fn perf_add(slot: PerfSlot, ns: u64, bytes: u64) {
    let s = &SLOTS[slot as usize];
    s.ns.fetch_add(ns, Relaxed);
    s.bytes.fetch_add(bytes, Relaxed);
    s.calls.fetch_add(1, Relaxed);
}

/// An RAII scope: time from construction to drop goes to `slot`, with
/// `bytes` of traffic.
pub(crate) struct PerfScope {
    slot: PerfSlot,
    bytes: u64,
    t0: Option<Instant>,
}

impl PerfScope {
    #[inline(always)]
    pub fn new(slot: PerfSlot, bytes: u64) -> PerfScope {
        PerfScope {
            slot,
            bytes,
            t0: if perf_enabled() { Some(Instant::now()) } else { None },
        }
    }
}

impl Drop for PerfScope {
    #[inline(always)]
    fn drop(&mut self) {
        if let Some(t0) = self.t0
            && perf_enabled()
        {
            perf_add(self.slot, t0.elapsed().as_nanos() as u64, self.bytes);
        }
    }
}

/// Turn the process-wide phase profiler on or off. Off by default; when on,
/// every kernel phase accumulates wall time and weight traffic.
pub fn perf_enable(enabled: bool) {
    perf_set(enabled);
}

/// Zero the profiler's totals.
pub fn perf_reset() {
    for s in &SLOTS {
        s.ns.store(0, Relaxed);
        s.bytes.store(0, Relaxed);
        s.calls.store(0, Relaxed);
    }
}

/// A table of the time and memory bandwidth per phase since the last
/// reset; `n_tokens` adds a per-token line when positive.
pub fn perf_report(title: &str, n_tokens: usize) -> String {
    let mut out = String::new();
    let mut total_ns = 0u64;
    let mut total_bytes = 0u64;
    for s in &SLOTS {
        total_ns += s.ns.load(Relaxed);
        total_bytes += s.bytes.load(Relaxed);
    }
    if total_ns == 0 {
        let _ = writeln!(out, "nncpu perf ({title}): no samples");
        return out;
    }

    let _ = writeln!(out, "nncpu perf — {title}");
    let _ = writeln!(
        out,
        "  {:<12} {:>10} {:>7} {:>10} {:>10} {:>10}",
        "phase", "ms", "%", "calls", "us/call", "GB/s"
    );
    let _ = writeln!(
        out,
        "  {:<12} {:>10} {:>7} {:>10} {:>10} {:>10}",
        "------------", "----------", "-------", "----------", "----------", "----------"
    );
    for (i, s) in SLOTS.iter().enumerate() {
        let calls = s.calls.load(Relaxed);
        if calls == 0 {
            continue;
        }
        let ns = s.ns.load(Relaxed) as f64;
        let bytes = s.bytes.load(Relaxed);
        let _ = write!(
            out,
            "  {:<12} {:>10.1} {:>6.1}% {:>10} {:>10.1}",
            NAMES[i],
            ns / 1e6,
            100.0 * ns / total_ns as f64,
            calls,
            ns / 1e3 / calls as f64
        );
        if bytes > 0 && ns > 0.0 {
            let _ = write!(out, " {:>10.1}", bytes as f64 / ns);
        }
        out.push('\n');
    }
    let _ = writeln!(out, "  {:<12} {:>10.1} {:>6.1}%", "TOTAL", total_ns as f64 / 1e6, 100.0);
    if total_bytes > 0 {
        let _ = writeln!(
            out,
            "  weight traffic: {:.2} GB @ {:.1} GB/s effective",
            total_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            total_bytes as f64 / total_ns as f64
        );
    }
    if n_tokens > 0 {
        let _ = writeln!(
            out,
            "  {:.2} ms/token over {} tokens",
            total_ns as f64 / 1e6 / n_tokens as f64,
            n_tokens
        );
    }
    out
}
