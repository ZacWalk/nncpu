//! Sampling: `std::mt19937_64` and MSVC's `uniform_real_distribution<double>`,
//! so a seed gives the replies the C++ llmcpu gives on Windows, and the
//! top-k / top-p / min-p / temperature chain.

use super::SamplerParams;
use std::cmp::Ordering;

pub(crate) struct Mt19937_64 {
    mt: [u64; 312],
    idx: usize,
}

impl Mt19937_64 {
    const NN: usize = 312;
    const MM: usize = 156;
    const MATRIX_A: u64 = 0xB502_6F5A_A966_19E9;
    const UPPER: u64 = 0xFFFF_FFFF_8000_0000;
    const LOWER: u64 = 0x7FFF_FFFF;

    pub fn new(seed: u64) -> Self {
        let mut mt = [0u64; 312];
        mt[0] = seed;
        for i in 1..Self::NN {
            mt[i] = 6364136223846793005u64
                .wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 62))
                .wrapping_add(i as u64);
        }
        Mt19937_64 { mt, idx: Self::NN }
    }

    pub fn next_u64(&mut self) -> u64 {
        if self.idx >= Self::NN {
            for i in 0..Self::NN {
                let x = (self.mt[i] & Self::UPPER) | (self.mt[(i + 1) % Self::NN] & Self::LOWER);
                let xa = (x >> 1) ^ if x & 1 != 0 { Self::MATRIX_A } else { 0 };
                self.mt[i] = self.mt[(i + Self::MM) % Self::NN] ^ xa;
            }
            self.idx = 0;
        }
        let mut x = self.mt[self.idx];
        self.idx += 1;
        x ^= (x >> 29) & 0x5555_5555_5555_5555;
        x ^= (x << 17) & 0x71D6_7FFF_EDA6_0000;
        x ^= (x << 37) & 0xFFF7_EEE0_0000_0000;
        x ^= x >> 43;
        x
    }

    /// A uniform draw from [0, b), as MSVC's
    /// `uniform_real_distribution<double>(0, b)` makes it: the top 53 bits of
    /// one draw, scaled.
    pub fn uniform(&mut self, b: f64) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9007199254740992.0) * b
    }
}

/// A seed from the OS's randomness (std seeds its hash keys from it) and the
/// clock.
pub(crate) fn random_seed() -> u64 {
    use std::hash::BuildHasher;
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64);
    std::collections::hash_map::RandomState::new().hash_one(t)
}

/// "a ranks before b": the higher logit first, then the lower id.
#[inline]
fn before(a: &(f32, i32), b: &(f32, i32)) -> bool {
    a.0 > b.0 || (a.0 == b.0 && a.1 < b.1)
}

/// A logit as the sampler takes it: NaN, which only a corrupt model gives,
/// becomes -inf, so the ranking is a total order and NaN is never drawn.
#[inline]
fn candidate_logit(l: f32) -> f32 {
    if l.is_nan() { f32::NEG_INFINITY } else { l }
}

fn rank(a: &(f32, i32), b: &(f32, i32)) -> Ordering {
    if before(a, b) {
        Ordering::Less
    } else if before(b, a) {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}

/// A heap entry whose maximum is the weakest candidate kept, so the heap's
/// root is the one a stronger logit replaces.
struct Weakest((f32, i32));

impl PartialEq for Weakest {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}
impl Eq for Weakest {}
impl PartialOrd for Weakest {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Weakest {
    fn cmp(&self, o: &Self) -> Ordering {
        rank(&self.0, &o.0)
    }
}

/// Scratch reused across tokens.
#[derive(Default)]
pub(crate) struct SamplerScratch {
    cand: Vec<(f32, i32)>,
    prob: Vec<f32>,
}

/// top-k -> top-p -> min-p on the model's own distribution, then temperature
/// over what survives, then a draw. The same order as llama.cpp's default
/// chain, so settings carry over.
pub(crate) fn sample_from_logits(
    s: &mut SamplerScratch,
    rng: &mut Mt19937_64,
    logits: &[f32],
    p: &SamplerParams,
) -> i32 {
    let n_vocab = logits.len();
    let cand = &mut s.cand;
    cand.clear();
    let k = if p.top_k > 0 && p.top_k < n_vocab {
        p.top_k
    } else {
        n_vocab
    };
    if k < n_vocab {
        // One pass with a k-element heap: almost every logit loses to the
        // heap's floor on a single compare.
        let mut heap = std::collections::BinaryHeap::with_capacity(k);
        for (i, &l) in logits.iter().enumerate() {
            let l = candidate_logit(l);
            if heap.len() < k {
                heap.push(Weakest((l, i as i32)));
            } else if l > heap.peek().map_or(f32::NEG_INFINITY, |w| w.0.0) {
                heap.pop();
                heap.push(Weakest((l, i as i32)));
            }
        }
        cand.extend(heap.into_iter().map(|w| w.0));
    } else {
        cand.extend(logits.iter().enumerate().map(|(i, &l)| (candidate_logit(l), i as i32)));
    }
    cand.sort_unstable_by(rank);
    if cand.is_empty() || !cand[0].0.is_finite() {
        return cand.first().map_or(0, |c| c.1);
    }

    // Probabilities at temperature 1, for the truncation rules.
    let prob = &mut s.prob;
    prob.resize(cand.len(), 0.0);
    let top = cand[0].0;
    let mut sum = 0.0f64;
    for (pi, c) in prob.iter_mut().zip(cand.iter()) {
        *pi = (c.0 - top).exp();
        sum += *pi as f64;
    }
    let mut keep = cand.len();
    if p.top_p > 0.0 && p.top_p < 1.0 {
        let mut cum = 0.0f64;
        for (i, &pi) in prob.iter().enumerate() {
            cum += pi as f64 / sum;
            if cum >= p.top_p as f64 {
                keep = i + 1;
                break;
            }
        }
    }
    if p.min_p > 0.0 {
        let floor = p.min_p * prob[0];
        let mut i = 1;
        while i < keep && prob[i] >= floor {
            i += 1;
        }
        keep = i;
    }

    // Temperature over the survivors, then one uniform draw.
    let inv_t = 1.0 / p.temperature as f64;
    let mut total = 0.0f64;
    for (pi, c) in prob.iter_mut().zip(cand.iter()).take(keep) {
        *pi = (((c.0 - top) as f64) * inv_t).exp() as f32;
        total += *pi as f64;
    }
    let mut r = rng.uniform(total);
    for (pi, c) in prob.iter().zip(cand.iter()).take(keep) {
        r -= *pi as f64;
        if r <= 0.0 {
            return c.1;
        }
    }
    cand[keep - 1].1
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Values captured from MSVC's std::mt19937_64 and
    /// uniform_real_distribution<double>, the C++ library's sampler.
    #[test]
    fn matches_msvc() {
        let mut r = Mt19937_64::new(12345);
        assert_eq!(r.next_u64(), 6597103971274460346);
        assert_eq!(r.next_u64(), 7386862472818278521);
        assert_eq!(r.next_u64(), 12716877617435052285);
        let mut r = Mt19937_64::new(42);
        for want in [
            2.643044365340886,
            2.236609878491441,
            2.632508202618093,
            0.4769543927135297,
        ] {
            assert_eq!(r.uniform(3.5), want);
        }
    }

    #[test]
    fn truncation() {
        let logits = [1.0f32, 5.0, 3.0, 5.0, -2.0];
        let mut s = SamplerScratch::default();
        let mut rng = Mt19937_64::new(1);
        // top_k 1 keeps only the best: logit 5.0 at the lower id.
        let p = SamplerParams {
            temperature: 1.0,
            top_k: 1,
            top_p: 1.0,
            min_p: 0.0,
        };
        for _ in 0..10 {
            assert_eq!(sample_from_logits(&mut s, &mut rng, &logits, &p), 1);
        }
        // min_p keeps the two tied best, nothing else.
        let p = SamplerParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.5,
        };
        for _ in 0..50 {
            let t = sample_from_logits(&mut s, &mut rng, &logits, &p);
            assert!(t == 1 || t == 3);
        }
        // A tiny temperature is all but greedy.
        let p = SamplerParams {
            temperature: 1e-3,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
        };
        let logits = [1.0f32, 2.0, 7.0, 3.0];
        for _ in 0..10 {
            assert_eq!(sample_from_logits(&mut s, &mut rng, &logits, &p), 2);
        }
    }

    /// A corrupt model's NaN logits rank last and are never drawn, whether
    /// they reach the sort through the top-k heap or not.
    #[test]
    fn nan_logits() {
        let mut logits = vec![f32::NAN; 64];
        logits[5] = 1.0;
        logits[9] = 0.5;
        let mut s = SamplerScratch::default();
        let mut rng = Mt19937_64::new(3);
        for top_k in [0, 2, 40] {
            let p = SamplerParams {
                temperature: 1.0,
                top_k,
                top_p: 1.0,
                min_p: 0.0,
            };
            for _ in 0..50 {
                let t = sample_from_logits(&mut s, &mut rng, &logits, &p);
                assert!(t == 5 || t == 9, "top_k {top_k}: drew {t}");
            }
        }
    }
}
