//! `cart` CLI. Port of `src/main.c`: pick a policy, run one verbose episode.

use std::time::{SystemTime, UNIX_EPOCH};

use hermes_nn::env::Env;
use hermes_nn::policy::{Policy, RandomPolicy, RevPolicy};
use hermes_nn::runner::run_episode;

fn main() {
    // Seed from the clock so each run differs, like C's `srand(time(NULL))`.
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u32)
        .unwrap_or(1);

    let name = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "random".to_string());
    let mut policy: Box<dyn Policy> = match name.as_str() {
        "random" => Box::new(RandomPolicy::new(seed)),
        "rev" => Box::new(RevPolicy),
        other => {
            eprintln!("unknown policy: {other} (expected random|rev)");
            std::process::exit(1);
        }
    };

    let mut env = Env::new(seed ^ 0x9E37_79B9); // distinct stream from the policy
    run_episode(&mut env, policy.as_mut(), true);
}
