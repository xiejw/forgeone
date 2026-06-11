//! GRPO (Group Relative Policy Optimization) training of the network policy.
//!
//! Unlike [`crate::trainer_reinforce`], this trainer **ignores the env reward**
//! entirely. Instead it samples a *group* of rollouts from the current policy
//! and hands each whole rollout to an [`LlmJudge`], which returns a single
//! scalar reward for the trajectory. The group's rewards are standardized into
//! group-relative advantages — `(r − mean) / (std + ε)` — and each rollout's
//! advantage is attached to **every** step of that rollout before the gradient
//! step. So an action is reinforced only insofar as its rollout beat the others
//! sampled alongside it; the group mean is the baseline (no learned value head).
//!
//! [`RandomJudge`] is a placeholder judge that returns a random reward; swap in
//! a real LLM-backed scorer by implementing [`LlmJudge`].

use crate::base::EPISODES;
use crate::env::Env;
use crate::policy::{Action, NNPolicy, Policy};
use crate::rng::Rng;
use crate::runner::MAX_STEPS;

/// Number of GRPO updates; each update samples a whole group of rollouts. Kept
/// equal to REINFORCE's [`EPISODES`] so the two trainers run the same length.
pub const ITERATIONS: usize = EPISODES;
/// Rollouts sampled per update — the "group" the advantage is relative to.
pub const GROUP_SIZE: usize = 8;
/// SGD learning rate.
pub const LR: f32 = 0.05;

// === --- LLM judge ---------------------------------------------------- ===

/// A single observed step of a rollout, handed to the judge so it can score the
/// trajectory as a whole. Carries no reward — producing the reward is the
/// judge's job.
#[derive(Clone, Copy, Debug)]
pub struct RolloutStep {
    pub pos: f32,
    pub speed: f32,
    pub action: Action,
}

/// Scores an entire rollout, standing in for an LLM-as-judge. The returned
/// scalar is the only reward signal GRPO uses; the env's own reward is ignored.
pub trait LlmJudge {
    /// Returns a scalar reward for the whole `rollout`.
    fn judge(&mut self, rollout: &[RolloutStep]) -> f32;
}

/// A placeholder judge that returns a uniform random reward in `[0, 1)`,
/// ignoring the rollout's contents. Stands in until a real LLM scorer is wired
/// up behind [`LlmJudge`].
pub struct RandomJudge {
    rng: Rng,
}

impl RandomJudge {
    /// Builds a random judge driven by the caller-supplied (split) `rng`.
    pub fn new(rng: Rng) -> RandomJudge {
        RandomJudge { rng }
    }
}

impl LlmJudge for RandomJudge {
    fn judge(&mut self, _rollout: &[RolloutStep]) -> f32 {
        self.rng.next_f32()
    }
}

// === --- GRPO --------------------------------------------------------- ===

/// Rolls out one episode, recording `(obs, action)` per step. The env reward is
/// deliberately discarded — only the judge assigns reward in GRPO.
fn rollout(env: &mut Env, policy: &mut NNPolicy) -> Vec<RolloutStep> {
    let mut history = Vec::new();
    for _ in 0..MAX_STEPS {
        let (pos, speed) = env.obs();
        let action = policy.act(pos, speed);
        let step = env.step(action);
        history.push(RolloutStep {
            pos: pos as f32,
            speed: speed as f32,
            action,
        });
        if step.done {
            break;
        }
    }
    history
}

/// Group-relative advantages: standardize the group's rewards to
/// `(r − mean) / (std + ε)`. This z-score is GRPO's baseline — it replaces
/// REINFORCE's per-episode discounted return. With fewer than two rollouts there
/// is no group to be relative to, so every advantage is zero (no signal).
fn group_advantages(rewards: &[f32]) -> Vec<f32> {
    if rewards.len() < 2 {
        return vec![0.0; rewards.len()];
    }
    let n = rewards.len() as f32;
    let mean = rewards.iter().sum::<f32>() / n;
    let var = rewards.iter().map(|r| (r - mean) * (r - mean)).sum::<f32>() / n;
    let std = var.sqrt();
    rewards.iter().map(|r| (r - mean) / (std + 1e-8)).collect()
}

/// GRPO trainer: per update, sample a group of rollouts, score each with the
/// judge, and push the policy toward rollouts that beat their group's average.
pub struct GrpoTrainer {
    pub lr: f32,
    pub group_size: usize,
}

impl Default for GrpoTrainer {
    fn default() -> GrpoTrainer {
        GrpoTrainer {
            lr: LR,
            group_size: GROUP_SIZE,
        }
    }
}

impl GrpoTrainer {
    /// Trains `policy` in place for `iterations` GRPO updates, scoring rollouts
    /// with `judge`. The caller-supplied (split) `rng` seeds an independent wind
    /// stream per rollout. Returns the final exponential moving average of the
    /// per-group mean judge reward (a training diagnostic).
    pub fn run(
        &self,
        policy: &mut NNPolicy,
        judge: &mut dyn LlmJudge,
        iterations: usize,
        rng: Rng,
    ) -> f32 {
        let mut seeder = rng;
        let mut avg_reward = 0.0;
        for it in 0..iterations {
            // 1. Sample a group of rollouts from the *current* policy; score each
            //    whole trajectory with the judge (env reward ignored).
            let mut group: Vec<Vec<RolloutStep>> = Vec::with_capacity(self.group_size);
            let mut rewards: Vec<f32> = Vec::with_capacity(self.group_size);
            for _ in 0..self.group_size {
                let mut env = Env::new(seeder.split());
                let history = rollout(&mut env, policy);
                rewards.push(judge.judge(&history));
                group.push(history);
            }

            // 2. Group-relative advantage, shared by every step of its rollout.
            let advantages = group_advantages(&rewards);

            // 3. Accumulate Σ over the whole group of ∇(−log π(a|s)·A), then one
            //    SGD step. Dividing by the group's total step count keeps the
            //    update scale independent of how long the rollouts ran.
            policy.nn.zero_grad();
            let mut total_steps = 0usize;
            for (history, &adv) in group.iter().zip(advantages.iter()) {
                for s in history {
                    policy.nn.accumulate(s.pos, s.speed, s.action, adv);
                }
                total_steps += history.len();
            }
            policy.nn.sgd_step(self.lr / total_steps.max(1) as f32);

            let mean_reward = rewards.iter().sum::<f32>() / rewards.len() as f32;
            avg_reward = if it == 0 {
                mean_reward
            } else {
                0.95 * avg_reward + 0.05 * mean_reward
            };
        }
        avg_reward
    }
}

// === --- Tests -------------------------------------------------------- ===

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_advantages_zero_mean() {
        let adv = group_advantages(&[1.0, 2.0, 3.0]);
        let mean: f32 = adv.iter().sum::<f32>() / adv.len() as f32;
        assert!(mean.abs() < 1e-5, "advantages not centered: {adv:?}");
        // Higher reward => higher (more positive) advantage.
        assert!(adv[0] < 0.0 && adv[2] > 0.0, "not ordered: {adv:?}");
    }

    #[test]
    fn group_advantages_degenerate_groups() {
        // Fewer than two rollouts: no group, so no signal.
        assert_eq!(group_advantages(&[]), Vec::<f32>::new());
        assert_eq!(group_advantages(&[5.0]), vec![0.0]);
        // All-equal rewards standardize to ~0 (nothing beat the average).
        let adv = group_advantages(&[2.0, 2.0, 2.0]);
        assert!(adv.iter().all(|a| a.abs() < 1e-3), "expected ~0: {adv:?}");
    }

    #[test]
    fn random_judge_in_unit_range() {
        let mut judge = RandomJudge::new(Rng::new(123));
        for _ in 0..1000 {
            let r = judge.judge(&[]);
            assert!((0.0..1.0).contains(&r), "reward out of [0,1): {r}");
        }
    }

    // GRPO runs end to end on a few small groups without panicking and returns a
    // finite diagnostic. With a random judge there is no learnable signal, so we
    // do not assert reward improves — only that the loop is well-formed.
    #[test]
    fn grpo_run_smoke() {
        let mut policy = NNPolicy::new(Rng::new(7));
        let mut judge = RandomJudge::new(Rng::new(8));
        let trainer = GrpoTrainer {
            lr: LR,
            group_size: 4,
        };
        let avg = trainer.run(&mut policy, &mut judge, 5, Rng::new(1));
        assert!(avg.is_finite(), "diagnostic reward not finite: {avg}");
    }
}
