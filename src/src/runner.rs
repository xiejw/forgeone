//! The episode loop and ASCII track renderer. Port of `src/runner.c` / `runner.h`.

use crate::env::{Env, POSITION_LIMIT};
use crate::policy::Policy;

/// Maximum number of steps before the episode is force-terminated.
pub const MAX_STEPS: usize = 1000;

/// Half-width of the ASCII track in cells; the full track is `2*HALF + 1` cells.
const TRACK_HALF: i64 = 30;
/// Total cells between the `[` and `]` brackets.
const TRACK_CELLS: usize = (2 * TRACK_HALF + 1) as usize;

// === --- Episode loop ------------------------------------------------- ===

/// Whether [`run_episode`] prints each step, replacing a bare `bool` flag.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Verbosity {
    /// Run silently; only the returned reward is produced.
    Quiet,
    /// Print an ASCII track frame per step plus the final total.
    Verbose,
}

/// Runs one episode on a freshly built `env`. Returns the total reward. When
/// [`Verbosity::Verbose`], prints an ASCII track frame per step plus the final
/// total, matching `py/runner.py`.
pub fn run_episode(env: &mut Env, policy: &mut dyn Policy, verbosity: Verbosity) -> f64 {
    let verbose = verbosity == Verbosity::Verbose;
    let mut total_reward = 0.0;
    for step in 0..MAX_STEPS {
        let (pos, speed) = env.obs();
        let act = policy.act(pos, speed);

        let result = env.step(act);
        total_reward += result.reward;

        if verbose {
            let (next_pos, next_speed) = env.obs();
            print_frame(step + 1, next_pos, next_speed, act, env.wind_dir);
        }
        if result.done {
            if verbose {
                println!("game over");
            }
            break;
        }
    }
    if verbose {
        println!("total reward: {total_reward:.1}");
    }
    total_reward
}

// === --- Rendering ---------------------------------------------------- ===

fn wind_symbol(wind_dir: i32) -> &'static str {
    if wind_dir > 0 {
        "wind-->"
    } else if wind_dir < 0 {
        "<--wind"
    } else {
        "  ...  "
    }
}

/// Builds the `[----|----#----]` track string with the center marker `|` and the
/// cart `#` placed from `pos`.
fn render(pos: f64) -> String {
    let scale = TRACK_HALF as f64 / POSITION_LIMIT;
    let idx = ((pos * scale).round() as i64 + TRACK_HALF).clamp(0, 2 * TRACK_HALF) as usize;

    let mut cells = [b'-'; TRACK_CELLS];
    cells[TRACK_HALF as usize] = b'|';
    cells[idx] = b'#'; // overwrites the center marker when the cart is centered

    let mut s = String::with_capacity(TRACK_CELLS + 2);
    s.push('[');
    s.push_str(std::str::from_utf8(&cells).expect("track is ASCII"));
    s.push(']');
    s
}

fn print_frame(step: usize, pos: f64, speed: f64, act: crate::policy::Action, wind_dir: i32) {
    // ANSI: cyan for the wind indicator, then reset.
    println!(
        "step {step:3} {track} pos={pos:+6.1} speed={speed:+5.1} act={name:<5} \x1b[36m{wind}\x1b[0m",
        track = render(pos),
        name = act.name(),
        wind = wind_symbol(wind_dir),
    );
}

// === --- Tests -------------------------------------------------------- ===

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::{Action, RandomPolicy, RevPolicy};

    // A policy wrapper that counts how many decisions it makes, so the test can
    // relate step_count to reward (mirrors `counting_random`/`counting_rev`).
    struct Counting<'a> {
        inner: &'a mut dyn Policy,
        steps: usize,
    }

    impl Policy for Counting<'_> {
        fn act(&mut self, pos: f64, speed: f64) -> Action {
            self.steps += 1;
            self.inner.act(pos, speed)
        }
    }

    // Mirrors py/test_policy.py: reward must land in [1, MAX_STEPS]. Each
    // surviving step earns 1.0; the final game-over step earns 0.0. So if the
    // cap is hit, step_count == reward; otherwise step_count == reward + 1.
    fn check_policy(name: &str, policy: &mut dyn Policy) {
        let mut env = Env::new(0xC0FFEE);
        let mut counting = Counting {
            inner: policy,
            steps: 0,
        };
        let reward = run_episode(&mut env, &mut counting, Verbosity::Quiet);
        let steps = counting.steps;

        assert!(
            (1.0..=MAX_STEPS as f64).contains(&reward),
            "[{name}] reward {reward} out of [1, {MAX_STEPS}]"
        );
        assert!(
            (1..=MAX_STEPS).contains(&steps),
            "[{name}] step_count {steps} out of [1, {MAX_STEPS}]"
        );
        let reward_int = reward as usize;
        assert!(
            steps == reward_int || steps == reward_int + 1,
            "[{name}] step_count {steps} not in {{reward, reward+1}} ({reward_int})"
        );
    }

    #[test]
    fn random_policy_reward_bounds() {
        check_policy("random", &mut RandomPolicy::new(1234));
    }

    #[test]
    fn rev_policy_reward_bounds() {
        check_policy("rev", &mut RevPolicy);
    }
}
