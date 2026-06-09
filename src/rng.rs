//! Splittable deterministic PRNG (DotMix), shared across the crate.
//!
//! # The DotMix algorithm
//!
//! Each draw advances `seed` by a fixed per-generator coefficient `gamma`
//! (the "Dot"), using addition modulo `George = 2^64 + 13` ([`update`]), then
//! scrambles the result with a MurmurHash3-style finalizer (the "Mix",
//! [`mix64`]). Rather than keep a fixed table of coefficients per pedigree
//! level, `gamma` is itself produced by a length-1 DotMix: a gamma-seed is
//! advanced modulo the prime `Percy = 2^56 - 5` and mixed by [`mix56`], so a
//! fresh, well-spread `gamma` is available for free at every [`split`].
//!
//! [`split`]: Rng::split

// === --- Constants ---------------------------------------------------- ===

/// `Percy`: the prime modulus used when deriving `gamma` coefficients. A prime
/// just under `2^56` makes the gamma-seed advance reduce in a single subtraction.
const GAMMA_PRIME: u64 = (1u64 << 56) - 5;

/// Coefficient that advances the gamma-seed (the "gamma of gamma").
const GAMMA_GAMMA: u64 = 0x0028_1E2D_BA66_06F3;

/// `George = 2^64 + 13`: the seed advance is addition modulo this value. Only
/// the additive `13` needs handling explicitly; the `2^64` is the natural u64
/// wraparound. Also the floor for any `gamma` (see DotMix).
const GEORGE_OFFSET: u64 = 13;

/// MurmurHash3 finalizer multipliers and shift, used by [`mix64`]/[`mix56`].
const MIX_C1: u64 = 0xff51_afd7_ed55_8ccd;
const MIX_C2: u64 = 0xc4ce_b9fe_1a85_ec53;
const MIX_SHIFT: u32 = 33;

/// Keeps the gamma mixer ([`mix56`]) within 56 bits so a derived gamma-seed
/// stays below [`GAMMA_PRIME`].
const MIX56_MASK: u64 = 0x00FF_FFFF_FFFF_FFFF;

/// Unit-in-the-last-place for a [0, 1) float built from the top 53 bits.
const DOUBLE_ULP: f64 = 1.0 / ((1u64 << 53) as f64);
const FLOAT_ULP: f32 = 1.0 / ((1u64 << 53) as f32);

// === --- Generator ---------------------------------------------------- ===

/// A splittable DotMix generator. Cheap to [`split`](Rng::split) into
/// independent child streams, and its full state round-trips through
/// [`state`](Rng::state) / [`from_state`](Rng::from_state).
pub struct Rng {
    /// Current seed; advanced by `gamma` on every draw.
    seed: u64,
    /// Per-generator advance coefficient (`>= 13` by construction).
    gamma: u64,
    /// Gamma-seed handed to children on `split`, also the saved gamma state.
    next_gamma_seed: u64,
}

impl Rng {
    /// Seeds a generator from `seed` with the canonical (zero) gamma-seed.
    pub fn new(seed: u64) -> Rng {
        Rng::with_gamma(seed, 0)
    }

    /// Seeds a generator from `seed` and a specific `gamma_seed`, which must be
    /// below `Percy` ([`GAMMA_PRIME`]). The gamma-seed is advanced once and
    /// mixed into this generator's `gamma`; the advanced value is what children
    /// inherit on [`split`](Rng::split).
    pub fn with_gamma(seed: u64, gamma_seed: u64) -> Rng {
        assert!(
            gamma_seed < GAMMA_PRIME,
            "gamma_seed must be < Percy (2^56 - 5)"
        );
        let mut next = gamma_seed.wrapping_add(GAMMA_GAMMA);
        if next >= GAMMA_PRIME {
            next -= GAMMA_PRIME;
        }
        Rng {
            seed,
            gamma: mix56(next).wrapping_add(GEORGE_OFFSET),
            next_gamma_seed: next,
        }
    }

    /// Splits off an independent child stream. Advances this generator's seed
    /// once and pairs it with the inherited gamma-seed, so parent and child
    /// diverge into well-separated streams. Deterministic: identical parents
    /// split into identical children.
    pub fn split(&mut self) -> Rng {
        let seed = self.advance_seed();
        Rng::with_gamma(seed, self.next_gamma_seed)
    }

    /// Captures the full generator state (`rng64To`) for later restore.
    pub fn state(&self) -> [u64; 2] {
        [self.seed, self.next_gamma_seed]
    }

    /// Rebuilds a generator from saved [`state`](Rng::state) (`rng64From`).
    pub fn from_state(state: [u64; 2]) -> Rng {
        Rng {
            seed: state[0],
            gamma: mix56(state[1]).wrapping_add(GEORGE_OFFSET),
            next_gamma_seed: state[1],
        }
    }
}

impl Rng {
    /// Draws a full 64-bit value: advance the seed (Dot), then finalize (Mix).
    pub fn next_u64(&mut self) -> u64 {
        mix64(self.advance_seed())
    }

    /// Draws a 32-bit value from the low half of [`next_u64`](Rng::next_u64).
    pub fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }

    /// Uniform `f64` in [0, 1), from the top 53 bits (the f64 mantissa width).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * DOUBLE_ULP
    }

    /// Uniform `f32` in [0, 1), from the top 53 bits.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 11) as f32 * FLOAT_ULP
    }

    /// Uniform `f64` in [lo, hi).
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + self.next_f64() * (hi - lo)
    }

    /// Either -1 or +1 with equal probability.
    pub fn sign(&mut self) -> i32 {
        if self.next_u64() & 1 == 1 { 1 } else { -1 }
    }

    /// Advances the seed in place by `gamma` (modulo George) and returns it.
    fn advance_seed(&mut self) -> u64 {
        self.seed = update(self.seed, self.gamma);
        self.seed
    }
}

// === --- DotMix primitives -------------------------------------------- ===

/// Adds `gamma` to `seed` modulo `George = 2^64 + 13`. The `seed + gamma` is
/// already reduced modulo `2^64`; the correction only matters when the true sum
/// overflowed (landed in `[2^64, 2^64 + 13)`), detected by `p < seed`. `gamma`
/// is constructed `> 13`, so the retry never loops.
fn update(seed: u64, gamma: u64) -> u64 {
    let p = seed.wrapping_add(gamma);
    if p >= seed {
        p
    } else if p >= GEORGE_OFFSET {
        p - GEORGE_OFFSET
    } else {
        p.wrapping_sub(GEORGE_OFFSET).wrapping_add(gamma)
    }
}

/// MurmurHash3 64-bit finalizer: scrambles a sequential seed into a uniformly
/// distributed value with good avalanche.
fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> MIX_SHIFT)).wrapping_mul(MIX_C1);
    z = (z ^ (z >> MIX_SHIFT)).wrapping_mul(MIX_C2);
    z ^ (z >> MIX_SHIFT)
}

/// As [`mix64`] but masked to 56 bits, so the result stays below `Percy` and is
/// a valid gamma-seed.
fn mix56(mut z: u64) -> u64 {
    z = ((z ^ (z >> MIX_SHIFT)).wrapping_mul(MIX_C1)) & MIX56_MASK;
    z = ((z ^ (z >> MIX_SHIFT)).wrapping_mul(MIX_C2)) & MIX56_MASK;
    z ^ (z >> MIX_SHIFT)
}

// === --- Tests -------------------------------------------------------- ===

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_for_same_seed() {
        let mut a = Rng::new(0xDEAD_BEEF);
        let mut b = Rng::new(0xDEAD_BEEF);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn distinct_seeds_diverge() {
        let mut a = Rng::new(1);
        let mut b = Rng::new(2);
        // Overwhelmingly likely to differ within a handful of draws.
        let differs = (0..8).any(|_| a.next_u64() != b.next_u64());
        assert!(differs);
    }

    #[test]
    fn split_is_deterministic_and_independent() {
        // Identical parents split into identical children.
        let mut p1 = Rng::new(42);
        let mut p2 = Rng::new(42);
        let mut c1 = p1.split();
        let mut c2 = p2.split();
        assert_eq!(c1.next_u64(), c2.next_u64());

        // Child and (advanced) parent are different streams.
        let child: Vec<u64> = (0..16).map(|_| c1.next_u64()).collect();
        let parent: Vec<u64> = (0..16).map(|_| p1.next_u64()).collect();
        assert_ne!(child, parent);
    }

    #[test]
    fn state_round_trips() {
        let mut rng = Rng::new(7);
        for _ in 0..5 {
            rng.next_u64();
        }
        let saved = rng.state();
        let expected: Vec<u64> = (0..10).map(|_| rng.next_u64()).collect();

        let mut restored = Rng::from_state(saved);
        let replayed: Vec<u64> = (0..10).map(|_| restored.next_u64()).collect();
        assert_eq!(expected, replayed);
    }

    #[test]
    fn from_state_clones_the_generator() {
        // After advancing, `from_state(rng.state())` must be an exact clone:
        // the captured state matches, and both produce the same draws in
        // lockstep going forward.
        let mut rng = Rng::new(0xABCD_1234);
        for _ in 0..17 {
            rng.next_u64();
        }

        let snapshot = rng.state();
        let mut clone = Rng::from_state(snapshot);
        assert_eq!(clone.state(), snapshot, "captured state did not round-trip");

        for _ in 0..1000 {
            assert_eq!(rng.next_u64(), clone.next_u64());
        }
    }

    #[test]
    fn floats_within_unit_interval() {
        let mut rng = Rng::new(123);
        for _ in 0..10_000 {
            let f = rng.next_f64();
            assert!((0.0..1.0).contains(&f));
            let g = rng.next_f32();
            assert!((0.0..1.0).contains(&g));
        }
    }

    #[test]
    fn range_and_sign_stay_in_bounds() {
        let mut rng = Rng::new(2024);
        for _ in 0..10_000 {
            let r = rng.range(-3.0, 5.0);
            assert!((-3.0..5.0).contains(&r));
            assert!(matches!(rng.sign(), -1 | 1));
        }
    }

    #[test]
    fn u32_is_low_half_of_u64() {
        // next_u32 must not just be a truncated re-draw; check it tracks the
        // low 32 bits of the same logical stream.
        let mut a = Rng::new(99);
        let mut b = Rng::new(99);
        for _ in 0..100 {
            assert_eq!(a.next_u32(), b.next_u64() as u32);
        }
    }
}
