//! Small deterministic xorshift PRNG, shared across the crate.
//!
//! Replaces C's global `srand`/`rand`: instead of one hidden global stream, each
//! consumer (the network's weight init, the env's wind, the random policy) owns
//! its own seeded [`Rng`]. The output does not bit-match C's `rand`; the ports
//! verify by behavior and bounds, not by reproducing the exact sequence.

/// A seeded xorshift32 generator.
pub struct Rng {
    state: u32,
}

impl Rng {
    /// Seeds the generator. The state is forced non-zero, since xorshift is
    /// stuck at zero.
    pub fn new(seed: u32) -> Rng {
        Rng { state: seed | 1 }
    }

    /// Raw 32-bit xorshift step.
    pub fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        x
    }

    /// Uniform `f32` in [0, 1), using the top 24 bits (the f32 mantissa width).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    /// Uniform `f64` in [0, 1), using all 32 bits.
    pub fn next_f64(&mut self) -> f64 {
        self.next_u32() as f64 / (1u64 << 32) as f64
    }

    /// Uniform `f64` in [lo, hi).
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + self.next_f64() * (hi - lo)
    }

    /// Either -1 or +1 with equal probability.
    pub fn sign(&mut self) -> i32 {
        if self.next_u32() & 1 == 1 {
            1
        } else {
            -1
        }
    }
}
