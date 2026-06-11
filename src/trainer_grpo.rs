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
//! [`LlamaJudge`] is the production judge — it asks a local llama.cpp server to
//! score the rollout. A test-only `RandomJudge` (in the `tests` module) stands
//! in where a deterministic, offline [`LlmJudge`] is needed.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

use crate::base::EPISODES;
use crate::env::Env;
use crate::policy::{Action, NNPolicy, Policy};
use crate::rng::Rng;
use crate::runner::MAX_STEPS;

/// Rollouts sampled per update — the "group" the advantage is relative to.
pub const GROUP_SIZE: usize = 8;
/// Number of GRPO updates. Each update samples [`GROUP_SIZE`] rollouts, so this
/// is [`EPISODES`] `/ GROUP_SIZE`: GRPO then consumes the same total number of
/// env rollouts as REINFORCE's [`EPISODES`] episodes, a fair sample budget.
pub const ITERATIONS: usize = EPISODES / GROUP_SIZE;
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

// === --- Llama judge -------------------------------------------------- ===

/// Default host/port of the llama.cpp server.
const DEFAULT_HOST: &str = "127.0.0.1";
const DEFAULT_PORT: u16 = 8080;
/// Tokens the judge is allowed to generate — a float string is short.
const N_PREDICT: u32 = 16;
/// Socket read/write timeout: generation can be slow, so be generous.
const HTTP_TIMEOUT_SECS: u64 = 120;
/// Cap on the action string in the prompt, so long rollouts stay bounded.
const MAX_ACTION_CHARS: usize = 256;

/// A real LLM judge: it asks a llama.cpp server (its `/completion` endpoint) to
/// rate a rollout, and parses the float reward out of the model's reply.
///
/// The server is assumed reachable and to return a float string; any failure
/// (connection, malformed reply, no number) panics, since a silently-wrong
/// reward would corrupt training. Run the server with e.g.
/// `llama-server -m model.gguf --port 8080`.
pub struct LlamaJudge {
    host: String,
    port: u16,
    n_predict: u32,
}

impl LlamaJudge {
    /// Targets the default local endpoint, `127.0.0.1:8080`.
    pub fn new() -> LlamaJudge {
        LlamaJudge::with_endpoint(DEFAULT_HOST, DEFAULT_PORT)
    }

    /// Targets an explicit `host:port` llama.cpp server.
    pub fn with_endpoint(host: &str, port: u16) -> LlamaJudge {
        LlamaJudge {
            host: host.to_string(),
            port,
            n_predict: N_PREDICT,
        }
    }
}

impl Default for LlamaJudge {
    fn default() -> LlamaJudge {
        LlamaJudge::new()
    }
}

impl LlmJudge for LlamaJudge {
    fn judge(&mut self, rollout: &[RolloutStep]) -> f32 {
        let body = build_request_body(&build_prompt(rollout), self.n_predict);
        let response = http_post(&self.host, self.port, "/completion", &body)
            .expect("llama.cpp judge request failed (is the server on :8080 up?)");
        let content = extract_string_field(http_body(&response), "content")
            .expect("llama.cpp response had no 'content' field");
        parse_leading_float(&content).expect("llama.cpp judge did not return a float reward")
    }
}

/// Builds the scoring prompt from a rollout: a short summary plus the action
/// sequence (capped), ending with an instruction to reply with one number.
fn build_prompt(rollout: &[RolloutStep]) -> String {
    let actions: String = rollout
        .iter()
        .take(MAX_ACTION_CHARS)
        .map(|s| match s.action {
            Action::None => 'n',
            Action::Left => 'l',
            Action::Right => 'r',
        })
        .collect();
    let (final_pos, final_speed) = rollout
        .last()
        .map(|s| (s.pos, s.speed))
        .unwrap_or((0.0, 0.0));
    format!(
        "You are an impartial judge scoring how well an agent balanced a cart.\n\
         The agent survived {steps} steps; final position {final_pos:.2}, final speed {final_speed:.2}.\n\
         Action sequence (n=none, l=left, r=right): {actions}\n\
         Reply with ONLY a single number between 0 and 1 rating the rollout's quality.",
        steps = rollout.len(),
    )
}

/// Wraps the prompt in llama.cpp's `/completion` JSON request body.
fn build_request_body(prompt: &str, n_predict: u32) -> String {
    format!(
        "{{\"prompt\":\"{}\",\"n_predict\":{n_predict},\"temperature\":0,\"stream\":false}}",
        json_escape(prompt),
    )
}

/// Escapes `s` for embedding inside a JSON string literal.
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// Sends a JSON POST and returns the raw HTTP response (headers + body). Uses
/// `Connection: close` so the body can be read to EOF without parsing framing.
fn http_post(host: &str, port: u16, path: &str, body: &str) -> std::io::Result<String> {
    let mut stream = TcpStream::connect((host, port))?;
    let timeout = Duration::from_secs(HTTP_TIMEOUT_SECS);
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    let request = format!(
        "POST {path} HTTP/1.1\r\n\
         Host: {host}:{port}\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {len}\r\n\
         Connection: close\r\n\
         \r\n\
         {body}",
        len = body.len(),
    );
    stream.write_all(request.as_bytes())?;
    stream.flush()?;
    let mut buf = Vec::new();
    stream.read_to_end(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

/// Returns the body of an HTTP response — everything after the blank line that
/// separates headers from body. Falls back to the whole string if absent.
fn http_body(response: &str) -> &str {
    match response.find("\r\n\r\n") {
        Some(i) => &response[i + 4..],
        None => response,
    }
}

/// Extracts the value of a JSON string field `"key": "…"` from `text`, decoding
/// the standard escapes. Returns `None` if the field or its string value is
/// missing.
fn extract_string_field(text: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let key_pos = text.find(&needle)?;
    let after_key = &text[key_pos + needle.len()..];
    let colon = after_key.find(':')?;
    let value = after_key[colon + 1..].trim_start();

    let chars: Vec<char> = value.chars().collect();
    if chars.first() != Some(&'"') {
        return None;
    }
    let mut out = String::new();
    let mut i = 1;
    while i < chars.len() {
        match chars[i] {
            '"' => return Some(out),
            '\\' => {
                i += 1;
                match *chars.get(i)? {
                    '"' => out.push('"'),
                    '\\' => out.push('\\'),
                    '/' => out.push('/'),
                    'n' => out.push('\n'),
                    'r' => out.push('\r'),
                    't' => out.push('\t'),
                    'b' => out.push('\u{08}'),
                    'f' => out.push('\u{0c}'),
                    'u' => {
                        let hex: String = chars.get(i + 1..i + 5)?.iter().collect();
                        let code = u32::from_str_radix(&hex, 16).ok()?;
                        out.push(char::from_u32(code).unwrap_or('\u{fffd}'));
                        i += 4;
                    }
                    _ => return None,
                }
            }
            c => out.push(c),
        }
        i += 1;
    }
    None
}

/// Parses the first floating-point number appearing in `s`. Tolerates the model
/// wrapping the number in stray text (e.g. `"reward: 0.8"`).
fn parse_leading_float(s: &str) -> Option<f32> {
    let trimmed = s.trim();
    if let Ok(v) = trimmed.parse::<f32>() {
        return Some(v);
    }
    let bytes = trimmed.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if c == b'-' || c == b'+' || c == b'.' || c.is_ascii_digit() {
            let start = i;
            i += 1;
            while i < bytes.len()
                && (bytes[i].is_ascii_digit()
                    || matches!(bytes[i], b'.' | b'e' | b'E' | b'+' | b'-'))
            {
                i += 1;
            }
            if let Ok(v) = trimmed[start..i].parse::<f32>() {
                return Some(v);
            }
        } else {
            i += 1;
        }
    }
    None
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

    /// A deterministic, offline judge for tests: returns a uniform random reward
    /// in `[0, 1)`, ignoring the rollout. Lets the GRPO loop be exercised without
    /// a live llama.cpp server.
    struct RandomJudge {
        rng: Rng,
    }

    impl RandomJudge {
        fn new(rng: Rng) -> RandomJudge {
            RandomJudge { rng }
        }
    }

    impl LlmJudge for RandomJudge {
        fn judge(&mut self, _rollout: &[RolloutStep]) -> f32 {
            self.rng.next_f32()
        }
    }

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

    #[test]
    fn json_escape_escapes_specials() {
        assert_eq!(json_escape("a\"b\\c\nd"), "a\\\"b\\\\c\\nd");
    }

    #[test]
    fn extract_content_from_llama_response() {
        // A trimmed-down shape of llama.cpp's /completion reply.
        let resp = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n\
                    {\"content\":\"0.83\",\"stop\":true,\"model\":\"x\"}";
        let body = http_body(resp);
        let content = extract_string_field(body, "content").expect("has content");
        assert_eq!(content, "0.83");
        assert_eq!(parse_leading_float(&content), Some(0.83));
    }

    #[test]
    fn extract_content_decodes_escapes() {
        let body = "{\"content\":\"  0.5\\n\",\"stop\":true}";
        let content = extract_string_field(body, "content").expect("has content");
        assert_eq!(content, "  0.5\n");
        assert_eq!(parse_leading_float(&content), Some(0.5));
    }

    #[test]
    fn parse_leading_float_handles_stray_text() {
        assert_eq!(parse_leading_float("0.7"), Some(0.7));
        assert_eq!(parse_leading_float("  reward: -1.25 out of 1"), Some(-1.25));
        assert_eq!(parse_leading_float("score is 3"), Some(3.0));
        assert_eq!(parse_leading_float("no number here"), None);
    }

    #[test]
    fn request_body_is_well_formed() {
        let body = build_request_body("hi \"there\"", 16);
        assert!(body.contains("\"prompt\":\"hi \\\"there\\\"\""));
        assert!(body.contains("\"n_predict\":16"));
        assert!(body.contains("\"stream\":false"));
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
