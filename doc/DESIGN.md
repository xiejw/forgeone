# hermes `nn` — Stack-VM Neural-Network Policy

## Context

`hermes` (the C port under `src/`) ships `RandomPolicy` and `RevPolicy`.
`src/CLAUDE.md` lists the next milestone as an `hermes_nn` engine (tensors,
linear, tanh, categorical, autograd) plus an `NNPolicy` + REINFORCE trainer,
deferred "until a torch-like engine lands."

`nn.h` / `nn.c` land the first piece: a small MLP policy whose **parameters are
statically sized struct members** and whose **forward** and **forward+backward
(training)** passes are expressed as two compiled **stack-VM instruction
streams** run by a single interpreter. It is intentionally **not** wired into
`main.c`, `runner.c`, or the `Makefile` build targets yet — only the files exist.

## Design decisions

- **Topology**: MLP `IN=2 (pos, speed) → HID=16 (tanh) → OUT=3 (action logits)`.
- **Objective**: REINFORCE policy gradient — loss `= -log π(a|s) · advantage`.
- **Training scope**: the training graph fills gradient buffers; a small
  `hermes_nn_sgd_step` applies them. Adam stays a follow-up.
- **Action selection**: softmax → **stochastic categorical sampling**.
- **Numeric type**: `double` (matches `env` / `policy`).
- **Error handling**: shapes are fully static, so internal invariants use
  `assert()`; no `forge_err_stack` is introduced (the project does not use it
  yet). Follows the in-tree style: `-std=c23`, spaced parens `( … )`, `hermes_`
  prefix, `_out` output params, section banners.

## Architecture

### Static parameter struct (`struct hermes_nn`)

All weights/biases and their gradient accumulators are fixed-size arrays (no
heap). Weight matrices are row-major: `w1[h * IN + i]` is the weight from input
`i` to hidden unit `h`.

```c
#define HERMES_NN_IN   2    // observation dims: position, speed
#define HERMES_NN_HID  16   // hidden units
#define HERMES_NN_OUT  3    // action logits: none / left / right

struct hermes_nn {
    double w1[HID * IN]; double b1[HID];   // layer 1
    double w2[OUT * HID]; double b2[OUT];  // layer 2
    double g_w1[...]; double g_b1[...];     // gradient accumulators,
    double g_w2[...]; double g_b2[...];     //   same shapes
    struct nn_program forward;   // logits + softmax only
    struct nn_program train;     // forward followed by backward
};
```

### The stack VM

A single interpreter, `nn_vm_run`, executes a program against a VM state. The
operand stack holds **vectors only**; matrix parameters are referenced by id and
pulled from the `hermes_nn` struct by the op. Reverse-mode autograd uses a
**LIFO tape**: each forward op saves the one activation its backward partner
needs, and backward pops them in reverse order — which lines up exactly.

```c
struct nn_val { double v[HERMES_NN_MAX_DIM]; int len; };  // MAX_DIM = HID

struct nn_vm {
    struct hermes_nn *nn;          // params + grad buffers
    struct nn_val stack[...]; int sp;   // operand stack
    struct nn_val tape[...];  int tp;   // saved forward activations (LIFO)
    const double *input;           // observation, length IN
    int    action;                 // chosen action (REINFORCE seed)
    double advantage;              // scalar reward signal
    double logp;                   // log π(action), set by DSOFTMAX
    double probs[HERMES_NN_OUT];   // softmax output, set by SOFTMAX
};
```

**Opcodes** (`enum nn_op`); each instruction is `{ op, a, b, c }` small ints,
with `a`/`b` typically parameter ids (`enum nn_param`) and `c` an output length.

Forward ops (push result on the value stack, save to tape for the partner):

| Op | Effect | Tape save |
|----|--------|-----------|
| `NN_OP_INPUT`   | push observation vector (len IN)       | — |
| `NN_OP_LINEAR`  | pop `x`, push `W·x + b`                 | input `x` |
| `NN_OP_TANH`    | pop `z`, push `tanh(z)` elementwise     | output `h` |
| `NN_OP_SOFTMAX` | pop logits, push probs; fills `probs`   | output `p` |

Backward ops (training program only; pop tape, accumulate into `g_*`):

| Op | Effect |
|----|--------|
| `NN_OP_DSOFTMAX_REINFORCE` | seed `dL/dz = (p − onehot(action))·advantage`; set `logp = log p[action]` |
| `NN_OP_DLINEAR` | given grad `g`, input `x`: `g_W += outer(g, x)`, `g_b += g`; push `Wᵀ·g` |
| `NN_OP_DTANH`   | `grad *= 1 − h²` using the taped tanh output |

#### Why the tape lines up

Forward execution saves, in order: `x0` (LINEAR1 input), `h1` (TANH output),
`h1` (LINEAR2 input), `p` (SOFTMAX output). Backward runs the reverse op order
and pops:

```
DSOFTMAX → p   (top)
DLINEAR2 → h1  (its input)
DTANH    → h1  (its output, for 1 − h²)
DLINEAR1 → x0  (its input)
```

Because each forward op pushes exactly what its backward partner consumes, the
tape behaves as a clean LIFO and no explicit indexing is needed.

### Compilation — `hermes_nn_compile`

Emits the two instruction streams from the fixed topology (idempotent; call once
after `init`):

- **forward**: `INPUT, LINEAR(w1,b1), TANH, LINEAR(w2,b2), SOFTMAX`.
- **train**: the forward prefix, then the reverse:
  `DSOFTMAX_REINFORCE, DLINEAR(w2,b2), DTANH, DLINEAR(w1,b1)`.

Storing programs as data (not hard-coded loops) is the point: the same VM runs
both, and a deeper network is added by emitting more instructions, not
rewriting the forward/backward math.

## Public API (`nn.h`)

```c
void  hermes_nn_init( struct hermes_nn *nn, unsigned seed );   // small random params, zero grads
void  hermes_nn_compile( struct hermes_nn *nn );               // build forward + train programs

// Forward: fill probs_out[OUT]; return a sampled action (stochastic categorical).
enum hermes_action hermes_nn_act( struct hermes_nn *nn, double pos, double speed,
                                  double probs_out[HERMES_NN_OUT] );

// Training: run the train program for one (obs, action, advantage) sample,
// accumulating into the g_* buffers; returns log π(action). Does not update params.
double hermes_nn_accumulate( struct hermes_nn *nn, double pos, double speed,
                             enum hermes_action action, double advantage );

void  hermes_nn_zero_grad( struct hermes_nn *nn );
void  hermes_nn_sgd_step( struct hermes_nn *nn, double lr );   // p -= lr * g
```

Internal (`static` in `nn.c`): `nn_vm_run`, per-op helpers, parameter lookup
(`nn_weight_of` / `nn_bias_of` / `nn_bias_grad_of`), softmax, and
`nn_sample_categorical` (uses `rand()` seeded by `hermes_nn_init`).

## Verification

Nothing is wired into the build yet, so verification is done in isolation:

1. **Compiles under project flags** — zero warnings under
   `cc -std=c23 -Wall -Werror -pedantic -Wextra -Wfatal-errors -Wconversion`.
2. **Numeric gradient check** — for a fixed `(obs, action, advantage)`, every
   `g_*` entry from `hermes_nn_accumulate` matches a central finite difference
   of the loss to `max |analytic − numeric| ≈ 2.7e-10` (gate `< 1e-5`). This is
   the correctness gate for the backward graph.
3. **Sanity behavior** — `probs` sum to `1.0` and are all positive; repeated
   `accumulate` + `sgd_step` with a constant positive advantage on one action
   raises that action's probability (`0.33 → 0.99` over 200 steps).

## Follow-ups

- Wire `NNPolicy` into the CLI and a REINFORCE training loop. `hermes_policy_fn`
  takes only `(pos, speed)` with no context arg, so the `struct hermes_nn *`
  must be threaded through some other way.
- Adam optimizer alongside / in place of `hermes_nn_sgd_step`.
- Add `nn.c` to `Makefile` `LIB_SRC` and a `test.c` case once it is wired in.
