//! A small cart RL demo.
//!
//! Modules: [`env`] (cart physics), [`policy`] (actions + the hand-written
//! `RandomPolicy`/`RevPolicy`), [`runner`] (episode loop + ASCII render), [`nn`]
//! (the stack-VM MLP engine), [`trainer_reinforce`] (the learned `NNPolicy` + REINFORCE
//! trainer), [`trainer_grpo`] (a GRPO trainer driven by an `LlmJudge` reward),
//! [`rng`] (the shared PRNG), and [`base`] (shared binary helpers).
//!
//! The [`nn`] engine is a small stack VM: forward ops push activations on a value
//! stack and save them on a LIFO tape; backward ops consume gradients off the
//! stack, read taped activations, and accumulate into the `g_*` buffers
//! (REINFORCE policy gradient). Each value owns its data in a `TensorId`-indexed
//! registry on the VM, so the stack and tape carry plain indices.

pub mod base;
pub mod env;
pub mod nn;
pub mod policy;
pub mod rng;
pub mod runner;
pub mod simple_json;
pub mod trainer_grpo;
pub mod trainer_reinforce;
