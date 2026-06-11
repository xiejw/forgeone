//! `cart` CLI: pick a policy, run one verbose episode.

use hermes_rl::base::seeded_rng;
use hermes_rl::env::Env;
use hermes_rl::policy::{Policy, RandomPolicy, RevPolicy};
use hermes_rl::runner::{Verbosity, run_episode};
use hermes_rl::trainer_reinforce::{EPISODES, NNPolicy, ReinforceTrainer};

fn main() {
    // One root generator for the whole program; every consumer gets its own
    // independent stream via `split()` rather than a separately seeded RNG.
    let mut rng = seeded_rng();

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
