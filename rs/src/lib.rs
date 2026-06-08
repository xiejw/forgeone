//! Rust port of the C `hermes_nn` engine (`src/nn.c` / `src/nn.h`).
//!
//! Keeps the same abstraction as the C version: a small stack VM whose forward
//! ops push activations on a value stack and save them on a LIFO tape, and
//! whose backward ops consume gradients off the stack, read taped activations,
//! and accumulate into the `g_*` buffers (REINFORCE policy gradient). The one
//! structural change is how activations are tracked: instead of raw `float*`
//! handles into a self-owned arena (which fights Rust's borrow checker), each
//! value owns its data in a `TensorId`-indexed registry on the VM.

pub mod nn;
