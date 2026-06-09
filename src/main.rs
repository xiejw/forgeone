//! `cart` CLI: pick a policy, run one verbose episode.

use std::time::{SystemTime, UNIX_EPOCH};

use hermes_nn::env::Env;
use hermes_nn::policy::{Policy, RandomPolicy, RevPolicy};
use hermes_nn::rng::Rng;
use hermes_nn::runner::{Verbosity, run_episode};
use hermes_nn::train::{EPISODES, NNPolicy, ReinforceTrainer};

fn main() {
    // Seed from the clock so each run differs, like C's `srand(time(NULL))`.
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(1);

    // One root generator for the whole program; every consumer gets its own
    // independent stream via `split()` rather than a separately seeded RNG.
    let mut rng = Rng::new(seed);

    let name = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "random".to_string());
    let mut policy: Box<dyn Policy> = match name.as_str() {
        "random" => Box::new(RandomPolicy::new(rng.split())),
        "rev" => Box::new(RevPolicy),
        "nn" => {
            // Train a fresh network with REINFORCE, then demo it.
            let mut p = NNPolicy::new(rng.split());
            let avg = ReinforceTrainer::default().run(&mut p, EPISODES, rng.split());
            eprintln!("[nn] trained {EPISODES} episodes; EMA reward ~ {avg:.1}");
            Box::new(p)
        }
        other => {
            eprintln!("unknown policy: {other} (expected random|rev|nn)");
            std::process::exit(1);
        }
    };

    let mut env = Env::new(rng.split()); // its own wind stream, split from root
    run_episode(&mut env, policy.as_mut(), Verbosity::Verbose);
}
