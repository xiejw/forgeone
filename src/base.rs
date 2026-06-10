//! Small shared helpers used by the binaries.

use std::time::{SystemTime, UNIX_EPOCH};

use crate::rng::Rng;

/// Builds the program's root [`Rng`], seeded from the wall clock so each process
/// run differs (like C's `srand(time(NULL))`). Binaries split independent
/// streams off the result rather than seeding generators separately. Panics if
/// the clock is before the Unix epoch (a broken system clock).
pub fn seeded_rng() -> Rng {
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is before the Unix epoch")
        .as_nanos() as u64;
    Rng::new(seed)
}
