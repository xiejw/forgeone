//! REINFORCE training of a network-backed policy. Port of `py/train.py`.
//!
//! [`NNPolicy`] wraps the stack-VM MLP ([`crate::nn::HermesNn`]) and implements
//! [`Policy`] by sampling from the softmax. [`ReinforceTrainer`] rolls out
//! episodes and nudges the policy so that actions followed by a high return
//! become more likely. Returns are discounted and standardized per episode as a
//! simple baseline that reduces gradient variance.
//!
//! The Python reference optimizes with Adam; this engine ships SGD
//! ([`HermesNn::sgd_step`]), so the update rule differs but the objective —
//! `loss = -Σ_t log π(a_t|s_t) · Ĝ_t` — is the same.

use crate::env::Env;
use crate::nn::HermesNn;
use crate::policy::{Action, Policy};
use crate::rng::Rng;
use crate::runner::MAX_STEPS;

/// Training episodes (mirrors `py/train.py`).
pub const EPISODES: usize = 400;
/// SGD learning rate.
pub const LR: f32 = 0.05;
/// Reward discount factor.
pub const GAMMA: f32 = 0.99;

// === --- NNPolicy ----------------------------------------------------- ===

/// A policy backed by the stack-VM MLP. Each decision samples from the network's
/// softmax over actions.
pub struct NNPolicy {
    nn: HermesNn,
}

impl NNPolicy {
    /// Wraps a fresh network seeded by the caller-supplied (split) `rng`.
    pub fn new(rng: Rng) -> NNPolicy {
        NNPolicy {
            nn: HermesNn::new(rng),
        }
    }
}

impl Policy for NNPolicy {
    fn act(&mut self, pos: f64, speed: f64) -> Action {
        // The env works in f64; the network in f32.
        self.nn.act(pos as f32, speed as f32).0
    }
}

// === --- REINFORCE ---------------------------------------------------- ===

/// One recorded step of a rollout: the observation, the sampled action, and the
/// immediate reward — everything the gradient step needs.
struct Transition {
    pos: f32,
    speed: f32,
    action: Action,
    reward: f32,
}

/// Rolls out one episode, recording the trajectory. Like `py/runner.py`'s
/// `one_off_run(verbose=False)`: returns `(total_reward, history)`.
fn rollout(env: &mut Env, policy: &mut NNPolicy) -> (f32, Vec<Transition>) {
    let mut history = Vec::new();
    let mut total = 0.0;
    for _ in 0..MAX_STEPS {
        let (pos, speed) = env.obs();
        let action = policy.act(pos, speed);
        let step = env.step(action);
        total += step.reward as f32;
        history.push(Transition {
            pos: pos as f32,
            speed: speed as f32,
            action,
            reward: step.reward as f32,
        });
        if step.done {
            break;
        }
    }
    (total, history)
}

/// `G_t = r_t + γ·r_{t+1} + …`, computed back-to-front.
fn discounted_returns(rewards: &[f32], gamma: f32) -> Vec<f32> {
    let mut out = vec![0.0; rewards.len()];
    let mut g = 0.0;
    for t in (0..rewards.len()).rev() {
        g = rewards[t] + gamma * g;
        out[t] = g;
    }
    out
}

/// Standardizes in place to `(x − mean) / (std + ε)` — the per-episode baseline.
/// A no-op for fewer than two elements (std is undefined / zero), matching the
/// Python `len(returns) > 1` guard.
fn standardize(xs: &mut [f32]) {
    if xs.len() < 2 {
        return;
    }
    let n = xs.len() as f32;
    let mean = xs.iter().sum::<f32>() / n;
    let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n;
    let std = var.sqrt();
    for x in xs.iter_mut() {
        *x = (*x - mean) / (std + 1e-8);
    }
}

/// REINFORCE trainer: roll out an episode, then push the policy toward actions
/// that preceded above-average return.
pub struct ReinforceTrainer {
    pub lr: f32,
    pub gamma: f32,
}

impl Default for ReinforceTrainer {
    fn default() -> ReinforceTrainer {
        ReinforceTrainer {
            lr: LR,
            gamma: GAMMA,
        }
    }
}

impl ReinforceTrainer {
    /// Trains `policy` in place for `episodes` episodes. The caller-supplied
    /// (split) `rng` is the seeder: each episode splits off a fresh, independent
    /// wind stream. Returns the final exponential moving average of episode
    /// reward.
    pub fn run(&self, policy: &mut NNPolicy, episodes: usize, rng: Rng) -> f32 {
        let mut seeder = rng;
        let mut avg_reward = 0.0;
        for ep in 0..episodes {
            let mut env = Env::new(seeder.split());
            let (reward, history) = rollout(&mut env, policy);

            let rewards: Vec<f32> = history.iter().map(|t| t.reward).collect();
            let mut returns = discounted_returns(&rewards, self.gamma);
            standardize(&mut returns);

            // Accumulate Σ_t ∇(−log π(a_t|s_t)·Ĝ_t), then one SGD step. Dividing
            // the step by the episode length keeps the update scale independent
            // of how long the episode ran (Adam does this implicitly upstream).
            policy.nn.zero_grad();
            for (t, &ret) in history.iter().zip(returns.iter()) {
                policy.nn.accumulate(t.pos, t.speed, t.action, ret);
            }
            policy.nn.sgd_step(self.lr / history.len().max(1) as f32);

            avg_reward = if ep == 0 {
                reward
            } else {
                0.95 * avg_reward + 0.05 * reward
            };
        }
        avg_reward
    }
}

// === --- Tests -------------------------------------------------------- ===

#[cfg(test)]
mod tests {
    use super::*;

    // Average reward over `runs` fresh episodes (sampling policy).
    fn eval_avg(policy: &mut NNPolicy, runs: usize, seed: u64) -> f32 {
        let mut seeder = Rng::new(seed);
        let mut total = 0.0;
        for _ in 0..runs {
            let mut env = Env::new(seeder.split());
            total += rollout(&mut env, policy).0;
        }
        total / runs as f32
    }

    #[test]
    fn discounted_returns_back_to_front() {
        // rewards [1, 1, 1], γ=0.5: G2=1, G1=1.5, G0=1.75.
        let g = discounted_returns(&[1.0, 1.0, 1.0], 0.5);
        assert_eq!(g, vec![1.75, 1.5, 1.0]);
    }

    #[test]
    fn standardize_zero_mean_unit_std() {
        let mut xs = vec![1.0, 2.0, 3.0];
        standardize(&mut xs);
        let mean: f32 = xs.iter().sum::<f32>() / 3.0;
        assert!(mean.abs() < 1e-5, "mean not ~0: {mean}");
        assert!(xs[0] < 0.0 && xs[2] > 0.0, "not centered: {xs:?}");
    }

    // Training raises evaluation reward well above the untrained network.
    #[test]
    fn training_improves_reward() {
        let untrained = {
            let mut p = NNPolicy::new(Rng::new(7));
            eval_avg(&mut p, 30, 999)
        };
        let trained = {
            let mut p = NNPolicy::new(Rng::new(7));
            ReinforceTrainer::default().run(&mut p, EPISODES, Rng::new(1));
            eval_avg(&mut p, 30, 999)
        };
        assert!(
            trained > untrained * 1.5,
            "training did not improve reward enough: {untrained:.1} -> {trained:.1}"
        );
    }
}
