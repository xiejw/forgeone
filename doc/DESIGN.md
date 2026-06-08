# `nn` — Stack-VM Neural-Network Policy

## Context

The crate (`src/`) is a small cart RL demo. The hand-written policies live in
`policy.rs` (`RandomPolicy`, `RevPolicy`), the physics in `env.rs`, the episode
loop and ASCII renderer in `runner.rs`, and the shared PRNG in `rng.rs`. This
document covers `nn.rs`: a fixed-topology MLP policy whose **forward** and
**forward+backward (training)** passes are expressed as two compiled **stack-VM
instruction streams** run by a single interpreter.

`nn` is self-contained and verified in isolation by its unit tests; it is **not**
yet wired into the `runner` (there is no `NNPolicy` — see Follow-ups). It began as
a port of a C implementation (now removed); the abstractions are preserved, with
a few Rust-shaped changes noted below.

## Design decisions

- **Topology**: MLP `IN=2 (pos, speed) → HID=16 (tanh) → OUT=3 (action logits)`.
- **Objective**: REINFORCE policy gradient — loss `= -log π(a|s) · reward`.
- **Training scope**: the training graph fills gradient buffers; `sgd_step`
  applies them (`p -= lr · g`). Adam is a follow-up.
- **Action selection**: softmax → **stochastic categorical sampling**.
- **Numeric type**: `f32` for the network (the env/policy use `f64`).
- **Invariants**: shapes are fully static, so the VM uses `assert!`/`expect`
  rather than an error type.
- **Activation storage**: each activation owns its data in a `TensorId`-indexed
  registry on the VM (`Vec<Vec<f32>>`), and the stack/tape carry indices. This
  replaces the C version's `float*` handles into a self-owned arena, which fights
  Rust's borrow checker.
- **Why a runtime bytecode VM and not type-level stack effects**: encoding each
  op's stack/tape effect in the type system would require replacing the runtime
  `Vec<Instr>` program with a compile-time typed composition (nightly
  `generic_const_exprs` or heavy trait machinery), discarding the
  "program as data" abstraction. For one fixed topology the runtime `assert`s are
  adequate, so the bytecode interpreter stays.

## Architecture

### Parameters + gradients (`struct Net`)

All weights/biases and their gradient accumulators are fixed-size arrays (no
heap). Weight matrices are row-major: `w1[h * IN + i]` is the weight from input
`i` to hidden unit `h`.

```rust
struct Net {
    w1: [f32; HID * IN], b1: [f32; HID],   // layer 1
    w2: [f32; OUT * HID], b2: [f32; OUT],  // layer 2
    g_w1: [...], g_b1: [...],              // gradient accumulators,
    g_w2: [...], g_b2: [...],              //   same shapes
}

pub struct HermesNn {
    net: Net,           // params + grad buffers (the VM's environment)
    forward: Program,   // logits + softmax only
    train: Program,     // forward followed by backward
    rng: Rng,
}
```

`Net` is kept **separate** from the compiled programs so the VM can borrow a
`&mut Net` (params/grads it writes) disjoint from a `&Program` (the stream it
reads) — two different fields of `HermesNn`, borrowed at once with no clone.

### The stack VM (`struct Vm`)

A single interpreter, `Vm::run`, executes a program. The VM holds only what it
runs against plus its scratch:

```rust
struct Vm<'a> {
    prog: &'a Program,      // the instruction stream (shared borrow)
    net:  &'a mut Net,      // params (read) + grad accumulators (written)
    tensors: Vec<Vec<f32>>, // activation registry; stack/tape index into it
    stack: Vec<TensorId>,   // operand stack (vectors only)
    tape:  Vec<TensorId>,   // saved forward activations (LIFO)
    input: [f32; IN],       // observation
    action: usize,          // chosen action (REINFORCE seed)
    reward: f32,            // scalar reward signal
    logp:  f32,             // log π(action), set by DSoftmaxReinforce
    probs: [f32; OUT],      // softmax output, set by Softmax
}
```

The operand stack holds **vectors only**; matrix parameters are referenced by id
(`enum Param`) and pulled from `net` by the op. Reverse-mode autograd uses a
**LIFO tape**: each forward op saves the one activation its backward partner
needs, and backward pops them in reverse order — which lines up exactly.

**Opcodes** (`enum Op`); each `Instr` is `{ op, a, b }` where `a`/`b` are the
parameter ids for Linear/DLinear (unused placeholders otherwise).

Forward ops (push result on the value stack, save to tape for the partner):

| Op | Effect | Tape save |
|----|--------|-----------|
| `Input`   | push observation vector (len IN)        | — |
| `Linear`  | pop `x`, push `W·x + b`                  | input `x` |
| `Tanh`    | pop `z`, push `tanh(z)` elementwise      | output `h` |
| `Softmax` | pop logits, push probs; fills `probs`    | output `p` |

Backward ops (training program only; pop tape, accumulate into `g_*`):

| Op | Effect |
|----|--------|
| `DSoftmaxReinforce` | seed `dL/dz = (p − onehot(action))·reward`; set `logp = log p[action]` |
| `DLinear` | given grad `g`, input `x`: `g_W += outer(g, x)`, `g_b += g`; push `Wᵀ·g` |
| `DTanh`   | `grad *= 1 − h²` using the taped tanh output |

#### Why the tape lines up

Forward execution saves, in order: `x0` (Linear1 input), `h1` (Tanh output),
`h1` (Linear2 input), `p` (Softmax output). Backward runs the reverse op order
and pops:

```
DSoftmaxReinforce → p   (top)
DLinear2          → h1  (its input)
DTanh             → h1  (its output, for 1 − h²)
DLinear1          → x0  (its input)
```

Because each forward op pushes exactly what its backward partner consumes, the
tape behaves as a clean LIFO and no explicit indexing is needed.

### Compilation — `HermesNn::compile`

Emits the two instruction streams from the fixed topology (idempotent; called by
`new`):

- **forward**: `Input, Linear(w1,b1), Tanh, Linear(w2,b2), Softmax`.
- **train**: the forward prefix, then the reverse:
  `DSoftmaxReinforce, DLinear(w2,b2), DTanh, DLinear(w1,b1)`.

Storing programs as data (not hard-coded loops) is the point: the same VM runs
both, and a deeper network is added by emitting more instructions, not rewriting
the forward/backward math.

## Public API (`nn.rs`)

```rust
impl HermesNn {
    pub fn new(seed: u32) -> HermesNn;          // small random params, zero grads, compile

    // Forward only: action probabilities for an observation (deterministic).
    pub fn forward_probs(&mut self, pos: f32, speed: f32) -> [f32; NN_OUT];

    // Forward + sample: returns a stochastic categorical action and its probs.
    pub fn act(&mut self, pos: f32, speed: f32) -> (Action, [f32; NN_OUT]);

    // Training: run the train program for one (obs, action, reward) sample,
    // accumulating into the g_* buffers; returns log π(action). Does not update params.
    pub fn accumulate(&mut self, pos: f32, speed: f32, action: Action, reward: f32) -> f32;

    pub fn zero_grad(&mut self);
    pub fn sgd_step(&mut self, lr: f32);        // p -= lr * g
}
```

`Action` is defined in `policy.rs` and re-exported from `nn`. Internal items are
private to the module: `Vm::run`, the per-op helpers, parameter lookup
(`Net::weight` / `Net::bias` / `Net::dlinear_refs`), softmax, and
`sample_categorical` (uses the `HermesNn`'s own `Rng`).

## Verification

`cargo test` (`nn::tests`) gates correctness:

1. **`grad_check`** — for a fixed `(obs, action, reward)`, every `g_*` entry from
   `accumulate` matches a central finite difference of the loss; gate
   `max |analytic − numeric| < 1e-3` (looser than the C `double` version because
   the network is `f32`). This is the correctness gate for the backward graph.
2. **`probs_are_distribution`** — `probs` sum to `1.0` and are all positive.
3. **`sgd_increases_chosen_prob`** — repeated `accumulate` + `sgd_step` with a
   constant positive reward on one action raises that action's probability.
4. **`determinism`** — equal seeds give equal initial weights and the same
   sampled action.

(The `policy` and `runner` modules carry their own tests: random-policy
uniformity, rev-policy steering, and episode reward bounds.)

## Follow-ups

- **Wire an `NNPolicy` into the runner.** This is now natural: implement the
  `policy::Policy` trait for a wrapper holding a `HermesNn` (converting the
  `f64` observation to `f32`), then run it through `run_episode`. Add a REINFORCE
  training loop on top.
- **Adam optimizer** alongside / in place of `sgd_step`.
