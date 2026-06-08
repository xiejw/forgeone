//! Actions and the hand-written policies.
//!
//! The `RandomPolicy`/`RevPolicy`  expressed as a [`Policy`] trait so a stateful policy (the random
//! one carries its own [`Rng`]) fits naturally.

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
    pub fn new(seed: u32) -> RandomPolicy {
        RandomPolicy {
            rng: Rng::new(seed),
        }
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
