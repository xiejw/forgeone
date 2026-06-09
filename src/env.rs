//! The cart environment physics.
//!
//! Each [`Env::step`] advances the simulation one tick: wind, the chosen action,
//! friction, motion, then the wind timer / restart roll. The cart survives until
//! `|position|` reaches [`POSITION_LIMIT`].

use crate::policy::Action;
use crate::rng::Rng;

// === --- Physics constants -------------------------------------------- ===

/// Cart ends the game when `|position|` reaches this many units.
pub const POSITION_LIMIT: f64 = 40.0;
/// Friction: speed moves toward zero by this much per second.
const SPEED_DECAY: f64 = 1.0;
/// Wind's initial strength (units/s) when a wind episode starts.
const WIND_ACCELERATION: f64 = 3.0;
/// Wind strength weakens this much per second, floored at 0.
const WIND_DECAY: f64 = 0.5;
/// Left/right action adds this many units/s to speed.
const ACTION_ACCELERATION: f64 = 2.0;
/// Probability that wind restarts after a wind episode ends.
const WIND_RESTART_CHANCE: f64 = 0.5;
/// Shortest a wind episode can last (seconds).
const WIND_MIN_DURATION: f64 = 1.0;
/// Longest a wind episode can last (seconds).
const WIND_MAX_DURATION: f64 = 5.0;
/// Simulation time step (seconds per action call).
const DT: f64 = 1.0;
/// Reward earned each surviving step.
const REWARD_PER_STEP: f64 = 1.0;

// === --- Env ---------------------------------------------------------- ===

/// Outcome of one [`Env::step`].
pub struct Step {
    /// Reward for this tick: [`REWARD_PER_STEP`] while alive, 0 on the step that
    /// ends the game.
    pub reward: f64,
    /// `true` on the step that transitions the env to game over.
    pub done: bool,
}

/// The cart state plus its own wind PRNG.
pub struct Env {
    pub position: f64,
    pub speed: f64,
    pub wind_dir: i32, // -1, 0, +1
    pub wind_remaining: f64,
    pub wind_strength: f64,
    pub game_over: bool,
    rng: Rng,
}

impl Env {
    /// A fresh env at the origin, at rest, with no wind. The caller passes the
    /// (already split) `rng` that drives this env's wind stream.
    pub fn new(rng: Rng) -> Env {
        Env {
            position: 0.0,
            speed: 0.0,
            wind_dir: 0,
            wind_remaining: 0.0,
            wind_strength: 0.0,
            game_over: false,
            rng,
        }
    }

    /// The observation fed to a policy: `(position, speed)`.
    pub fn obs(&self) -> (f64, f64) {
        (self.position, self.speed)
    }

    /// Advances the simulation by one tick. Calling on an already-over env is a
    /// programmer error.
    pub fn step(&mut self, act: Action) -> Step {
        assert!(!self.game_over);

        self.apply_wind();
        self.apply_action(act);
        self.apply_friction();
        self.position += self.speed * DT;
        self.tick_wind_timer();
        self.maybe_restart_wind();

        if self.position.abs() >= POSITION_LIMIT {
            self.game_over = true;
            Step {
                reward: 0.0,
                done: true,
            }
        } else {
            Step {
                reward: REWARD_PER_STEP,
                done: false,
            }
        }
    }

    // --- per-tick physics ---

    fn apply_wind(&mut self) {
        if self.wind_dir == 0 {
            return;
        }
        self.speed += f64::from(self.wind_dir) * self.wind_strength * DT;
        self.wind_strength -= WIND_DECAY * DT;
        if self.wind_strength < 0.0 {
            self.wind_strength = 0.0;
        }
    }

    fn apply_action(&mut self, act: Action) {
        match act {
            Action::Right => self.speed += ACTION_ACCELERATION * DT,
            Action::Left => self.speed -= ACTION_ACCELERATION * DT,
            Action::None => {}
        }
    }

    fn apply_friction(&mut self) {
        if self.speed > 0.0 {
            self.speed -= SPEED_DECAY * DT;
            if self.speed < 0.0 {
                self.speed = 0.0;
            }
        } else if self.speed < 0.0 {
            self.speed += SPEED_DECAY * DT;
            if self.speed > 0.0 {
                self.speed = 0.0;
            }
        }
    }

    fn tick_wind_timer(&mut self) {
        if self.wind_dir == 0 {
            return;
        }
        self.wind_remaining -= DT;
        if self.wind_remaining <= 0.0 {
            self.wind_dir = 0;
            self.wind_remaining = 0.0;
        }
    }

    fn maybe_restart_wind(&mut self) {
        if self.wind_dir != 0 {
            return;
        }
        if self.rng.next_f64() >= WIND_RESTART_CHANCE {
            return;
        }
        self.wind_dir = self.rng.sign();
        self.wind_strength = WIND_ACCELERATION;
        self.wind_remaining = self.rng.range(WIND_MIN_DURATION, WIND_MAX_DURATION);
    }
}
