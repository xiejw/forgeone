//! Actions and the policies.
//!
//! The hand-written `RandomPolicy`/`RevPolicy` and the network-backed
//! [`NNPolicy`] are expressed as a [`Policy`] trait so a stateful policy (the
//! random one carries its own [`Rng`], the network one its MLP) fits naturally.

use crate::nn::HermesNn;
use crate::rng::Rng;

/// The three discrete actions. Mirrors `enum hermes_action` from `policy.h`.
/// Discriminants match the network's softmax output order.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Action {
    None = 0,
    Left = 1,
    Right = 2,
}

impl Action {
    /// Maps an index (softmax output slot, or a `rand % 3` draw) to an action.
    pub fn from_index(i: usize) -> Action {
        match i {
            0 => Action::None,
            1 => Action::Left,
            2 => Action::Right,
            _ => panic!("action index out of range: {i}"),
        }
    }

    /// Human-readable name, for the runner's per-step display.
    pub fn name(self) -> &'static str {
        match self {
            Action::None => "none",
            Action::Left => "left",
            Action::Right => "right",
        }
    }
}

/// A decision rule mapping an observation `(pos, speed)` to an [`Action`].
/// `&mut self` lets a policy hold state (e.g. a PRNG) across steps.
pub trait Policy {
    fn act(&mut self, pos: f64, speed: f64) -> Action;
}

// === --- RandomPolicy ---------------------------------------------------- ===

/// Picks a uniformly random action each step, ignoring the observation.
pub struct RandomPolicy {
    rng: Rng,
}

impl RandomPolicy {
    /// Builds a random policy driven by the caller-supplied (split) `rng`.
    pub fn new(rng: Rng) -> RandomPolicy {
        RandomPolicy { rng }
    }
}

impl Policy for RandomPolicy {
    fn act(&mut self, _pos: f64, _speed: f64) -> Action {
        Action::from_index((self.rng.next_u32() % 3) as usize)
    }
}

// === --- Reverse Policy -------------------------------------------------- ===

/// Steers back toward the center: push left of center-right, right of
/// center-left, do nothing dead center.
pub struct RevPolicy;

impl Policy for RevPolicy {
    fn act(&mut self, pos: f64, _speed: f64) -> Action {
        if pos > 0.0 {
            Action::Left
        } else if pos < 0.0 {
            Action::Right
        } else {
            Action::None
        }
    }
}

// === --- NNPolicy ----------------------------------------------------- ===

/// A policy backed by the stack-VM MLP. Each decision samples from the network's
/// softmax over actions.
pub struct NNPolicy {
    /// Crate-visible so the trainers (e.g. [`crate::trainer_reinforce`],
    /// [`crate::trainer_grpo`]) can drive the gradient ops —
    /// `zero_grad`/`accumulate`/`sgd_step`.
    pub(crate) nn: HermesNn,
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

// === --- Tests -------------------------------------------------------- ===

#[cfg(test)]
mod tests {
    use super::*;

    // A) RandomPolicy is roughly uniform over the three actions: each should
    // appear within 10% of n/3. With n = 30_000 that band is ~12 sigma wide, so
    // the (seeded, deterministic) test is comfortable, not flaky.
    #[test]
    fn random_policy_is_roughly_uniform() {
        let n = 30_000;
        let mut policy = RandomPolicy::new(Rng::new(2024));
        let (mut none, mut left, mut right) = (0, 0, 0);
        for _ in 0..n {
            match policy.act(0.0, 0.0) {
                Action::None => none += 1,
                Action::Left => left += 1,
                Action::Right => right += 1,
            }
        }
        assert_eq!(none + left + right, n, "every draw is one of three actions");

        let expected = n as f64 / 3.0;
        let tol = expected * 0.10;
        for (name, count) in [("none", none), ("left", left), ("right", right)] {
            let diff = (count as f64 - expected).abs();
            assert!(
                diff <= tol,
                "{name}: count {count} too far from {expected:.0} (|diff| {diff:.0} > {tol:.0})"
            );
        }
    }

    // B) RevPolicy steers toward center: Left when right of center, Right when
    // left of center, None dead center — and it ignores speed.
    #[test]
    fn rev_policy_steers_toward_center() {
        let mut policy = RevPolicy;
        assert_eq!(
            policy.act(5.0, 0.0),
            Action::Left,
            "right of center -> left"
        );
        assert_eq!(
            policy.act(-5.0, 0.0),
            Action::Right,
            "left of center -> right"
        );
        assert_eq!(policy.act(0.0, 0.0), Action::None, "dead center -> none");

        // Speed must not change the decision.
        assert_eq!(policy.act(5.0, 100.0), Action::Left);
        assert_eq!(policy.act(5.0, -100.0), Action::Left);
        assert_eq!(policy.act(-5.0, 100.0), Action::Right);
        assert_eq!(policy.act(0.0, 100.0), Action::None);
    }
}
