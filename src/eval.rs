//! `eval` CLI: run every policy `RUNS` times and report the average reward.
//!
use hermes_rl::base::seeded_rng;
use hermes_rl::env::Env;
use hermes_rl::policy::{Policy, RandomPolicy, RevPolicy};
use hermes_rl::rng::Rng;
use hermes_rl::runner::{Verbosity, run_episode};
use hermes_rl::trainer_reinforce::{EPISODES, NNPolicy, ReinforceTrainer};

/// Episodes averaged per policy.
const RUNS: usize = 10;

/// Column headers for the results table.
const COL_POLICY: &str = "Policy";
const COL_REWARD: &str = "Avg Reward";

/// Runs `policy` for [`RUNS`] episodes, each on a fresh env split from
/// `env_rng`, and returns the mean total reward.
fn eval_policy(policy: &mut dyn Policy, env_rng: &mut Rng) -> f64 {
    let mut total = 0.0;
    for _ in 0..RUNS {
        let mut env = Env::new(env_rng.split());
        total += run_episode(&mut env, policy, Verbosity::Quiet);
    }
    total / RUNS as f64
}

/// Renders `(name, avg_reward)` rows as a bordered ASCII table, sizing each
/// column to its widest cell.
fn print_table(rows: &[(&str, f64)]) {
    let cells: Vec<(String, String)> = rows
        .iter()
        .map(|(name, avg)| (name.to_string(), format!("{avg:.1}")))
        .collect();

    let w_name = cells
        .iter()
        .map(|(n, _)| n.len())
        .chain([COL_POLICY.len()])
        .max()
        .unwrap_or(0);
    let w_reward = cells
        .iter()
        .map(|(_, r)| r.len())
        .chain([COL_REWARD.len()])
        .max()
        .unwrap_or(0);

    let border = format!("+-{}-+-{}-+", "-".repeat(w_name), "-".repeat(w_reward));
    println!("{border}");
    // Header: policy left-aligned, reward right-aligned (numbers read better).
    println!("| {COL_POLICY:<w_name$} | {COL_REWARD:>w_reward$} |");
    println!("{border}");
    for (name, reward) in &cells {
        println!("| {name:<w_name$} | {reward:>w_reward$} |");
    }
    println!("{border}");
}

fn main() {
    // One root generator; every consumer below gets its own stream via split().
    let mut rng = seeded_rng();

    // Train the network policy once before evaluating it.
    let mut nn = NNPolicy::new(rng.split());
    ReinforceTrainer::default().run(&mut nn, EPISODES, rng.split());

    let mut random = RandomPolicy::new(rng.split());
    let mut rev = RevPolicy;

    // (name, policy) pairs; each is evaluated on its own env stream.
    let mut policies: [(&str, &mut dyn Policy); 3] =
        [("random", &mut random), ("rev", &mut rev), ("nn", &mut nn)];

    let results: Vec<(&str, f64)> = policies
        .iter_mut()
        .map(|(name, policy)| (*name, eval_policy(*policy, &mut rng.split())))
        .collect();

    println!("Average reward over {RUNS} runs per policy:");
    print_table(&results);
}
