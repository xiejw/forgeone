//! Rust port of the C hermes cart RL demo (`src/`).
//!
//! Modules mirror the C files: [`env`] (physics, `env.c`), [`policy`] (actions +
//! hand-written policies, `policy.c`), [`runner`] (episode loop, `runner.c`),
//! [`nn`] (the stack-VM MLP, `nn.c`), and [`rng`] (shared PRNG, replacing C's
//! global `rand`).
//!
//! The [`nn`] engine keeps the same abstraction as the C version: a small stack
//! VM whose forward ops push activations on a value stack and save them on a
//! LIFO tape, and whose backward ops consume gradients off the stack, read taped
//! activations, and accumulate into the `g_*` buffers (REINFORCE policy
//! gradient). The one structural change is how activations are tracked: instead
//! of raw `float*` handles into a self-owned arena (which fights Rust's borrow
//! checker), each value owns its data in a `TensorId`-indexed registry on the VM.

pub mod env;
pub mod nn;
pub mod policy;
pub mod rng;
pub mod runner;
