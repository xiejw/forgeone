//! Fixed-topology MLP policy (2 -> 16 tanh -> 3 softmax) run by a tiny stack VM.
//!
//! Mirrors `src/nn.c`: parameters and gradient accumulators live inline in
//! [`HermesNn`]; `hermes_nn_compile` becomes [`HermesNn::compile`]; the forward
//! and training instruction streams are the same. The VM ([`Vm`]) holds a
//! `TensorId` registry of owned activations in place of the C scratch arena, so
//! the stack and tape carry plain indices instead of raw pointers.

// === --- Topology ----------------------------------------------------- ===

/// Observation dimensions fed to the network: position and speed.
pub const NN_IN: usize = 2;
/// Hidden units in the single tanh layer.
pub const NN_HID: usize = 16;
/// Output logits, one per action: none / left / right.
pub const NN_OUT: usize = 3;

/// Half-width of the uniform range for initial weights: w ~ U(-SCALE, SCALE).
const NN_INIT_SCALE: f32 = 0.1;

// === --- Actions ------------------------------------------------------ ===

/// Re-export of the action type, which lives in [`crate::policy`] (mirroring C's
/// `policy.h`). The softmax output maps onto it via [`Action::from_index`].
pub use crate::policy::Action;

// === --- Stack VM program model --------------------------------------- ===

/// Opcodes for the stack VM. Forward ops push their result on the value stack
/// and record it on the activation tape; backward ops consume gradients off the
/// stack, read taped activations, and accumulate into the `g_*` buffers.
#[derive(Clone, Copy)]
enum Op {
    // Forward (the comment notes what the op tapes for its backward partner).
    Input,   // push the observation vector (len NN_IN); tapes nothing
    Linear,  // pop x, push W*x + b; tapes x
    Tanh,    // pop z, push tanh(z); tapes the output
    Softmax, // pop logits, push softmax probs; tapes probs

    // Backward (training program only); each pops the matching taped value.
    DSoftmaxReinforce, // tape: probs; seed (probs - onehot(action)) * reward
    DTanh,             // tape: tanh(z); grad *= 1 - tanh(z)^2
    DLinear,           // tape: x; accumulate g_W, g_b; push W^T * grad
}

/// Identifies a parameter block (weight matrix or bias vector) inside
/// [`HermesNn`], so a Linear/DLinear instruction can reference it by id.
#[derive(Clone, Copy)]
enum Param {
    W1,
    B1,
    W2,
    B2,
}

/// A single VM instruction. `a`/`b` name the weight and bias blocks for
/// Linear/DLinear; for other ops they are unread placeholders.
#[derive(Clone, Copy)]
struct Instr {
    op: Op,
    a: Param,
    b: Param,
}

impl Instr {
    fn new(op: Op, a: Param, b: Param) -> Instr {
        Instr { op, a, b }
    }
}

/// A compiled instruction stream produced by [`HermesNn::compile`].
struct Program {
    instr: Vec<Instr>,
}

// === --- PRNG --------------------------------------------------------- ===

// The xorshift PRNG lives in [`crate::rng`]; weight init and categorical
// sampling use it.
use crate::rng::Rng;

// === --- Network ------------------------------------------------------ ===

/// Parameters and gradient accumulators for the network — the environment the
/// VM reads from and writes into. All fields are fixed-size arrays (no heap).
/// Weight matrices are row-major: `w1[h * NN_IN + i]` is the weight from input
/// `i` to hidden unit `h`. Kept separate from the compiled programs in
/// [`HermesNn`] so the VM can hold a `&mut Net` disjoint from a `&Program`.
struct Net {
    // Parameters.
    w1: [f32; NN_HID * NN_IN],
    b1: [f32; NN_HID],
    w2: [f32; NN_OUT * NN_HID],
    b2: [f32; NN_OUT],
    // Gradient accumulators, same shapes as the parameters above.
    g_w1: [f32; NN_HID * NN_IN],
    g_b1: [f32; NN_HID],
    g_w2: [f32; NN_OUT * NN_HID],
    g_b2: [f32; NN_OUT],
}

/// A fixed-topology MLP policy: the parameter/gradient environment ([`Net`])
/// plus the compiled stack-VM programs it runs.
pub struct HermesNn {
    net: Net,
    // Compiled stack-VM programs (filled by `compile`).
    forward: Program, // logits + softmax only
    train: Program,   // forward followed by backward
    rng: Rng,
}

impl HermesNn {
    /// Initializes parameters with small random values, zeroes the gradients,
    /// and compiles the forward/training programs. `seed` makes the result
    /// reproducible. (Combines C's `hermes_nn_init` + `hermes_nn_compile`.)
    pub fn new(seed: u32) -> HermesNn {
        let mut nn = HermesNn {
            net: Net {
                w1: [0.0; NN_HID * NN_IN],
                b1: [0.0; NN_HID],
                w2: [0.0; NN_OUT * NN_HID],
                b2: [0.0; NN_OUT],
                g_w1: [0.0; NN_HID * NN_IN],
                g_b1: [0.0; NN_HID],
                g_w2: [0.0; NN_OUT * NN_HID],
                g_b2: [0.0; NN_OUT],
            },
            forward: Program { instr: Vec::new() },
            train: Program { instr: Vec::new() },
            rng: Rng::new(seed),
        };
        for i in 0..nn.net.w1.len() {
            nn.net.w1[i] = (nn.rng.next_f32() * 2.0 - 1.0) * NN_INIT_SCALE;
        }
        for i in 0..nn.net.w2.len() {
            nn.net.w2[i] = (nn.rng.next_f32() * 2.0 - 1.0) * NN_INIT_SCALE;
        }
        nn.compile();
        nn
    }

    /// Builds the forward and training stack-VM programs from the fixed
    /// topology. Idempotent.
    fn compile(&mut self) {
        let fwd = vec![
            Instr::new(Op::Input, Param::W1, Param::B1),
            Instr::new(Op::Linear, Param::W1, Param::B1),
            Instr::new(Op::Tanh, Param::W1, Param::B1),
            Instr::new(Op::Linear, Param::W2, Param::B2),
            Instr::new(Op::Softmax, Param::W2, Param::B2),
        ];
        // The training program is the forward pass followed by its reverse.
        let mut trn = fwd.clone();
        trn.push(Instr::new(Op::DSoftmaxReinforce, Param::W2, Param::B2));
        trn.push(Instr::new(Op::DLinear, Param::W2, Param::B2));
        trn.push(Instr::new(Op::DTanh, Param::W2, Param::B2));
        trn.push(Instr::new(Op::DLinear, Param::W1, Param::B1));

        self.forward = Program { instr: fwd };
        self.train = Program { instr: trn };
    }

    /// Runs the forward program for the observation `(pos, speed)` and returns
    /// the action probabilities. Does not consume the PRNG (deterministic).
    pub fn forward_probs(&mut self, pos: f32, speed: f32) -> [f32; NN_OUT] {
        // `&self.forward` (shared) and `&mut self.net` (mut) are disjoint fields,
        // so the VM borrows both at once — no clone, no move.
        let mut vm = Vm::new(&self.forward, &mut self.net, [pos, speed]);
        vm.run();
        vm.probs
    }

    /// Runs the forward program, then samples and returns an action from the
    /// resulting categorical distribution along with the probabilities.
    pub fn act(&mut self, pos: f32, speed: f32) -> (Action, [f32; NN_OUT]) {
        let probs = self.forward_probs(pos, speed);
        let idx = self.sample_categorical(&probs);
        (Action::from_index(idx), probs)
    }

    /// Runs the training program for one REINFORCE sample, accumulating the
    /// policy gradient of `-log pi(action | obs) * reward` into the `g_*`
    /// buffers. Returns `log pi(action | obs)`. Does not modify parameters.
    pub fn accumulate(&mut self, pos: f32, speed: f32, action: Action, reward: f32) -> f32 {
        // See `forward_probs`: the program and the param/grad environment are
        // disjoint fields, so the VM borrows them together. No copy.
        let mut vm = Vm::new(&self.train, &mut self.net, [pos, speed]);
        vm.action = action as usize;
        vm.reward = reward;
        vm.run();
        vm.logp
    }

    /// Zeroes all gradient accumulators.
    pub fn zero_grad(&mut self) {
        self.net.g_w1 = [0.0; NN_HID * NN_IN];
        self.net.g_b1 = [0.0; NN_HID];
        self.net.g_w2 = [0.0; NN_OUT * NN_HID];
        self.net.g_b2 = [0.0; NN_OUT];
    }

    /// Applies a single SGD update `p -= lr * g` across all parameters.
    pub fn sgd_step(&mut self, lr: f32) {
        let net = &mut self.net;
        for (w, g) in net.w1.iter_mut().zip(net.g_w1.iter()) {
            *w -= lr * g;
        }
        for (w, g) in net.b1.iter_mut().zip(net.g_b1.iter()) {
            *w -= lr * g;
        }
        for (w, g) in net.w2.iter_mut().zip(net.g_w2.iter()) {
            *w -= lr * g;
        }
        for (w, g) in net.b2.iter_mut().zip(net.g_b2.iter()) {
            *w -= lr * g;
        }
    }

    fn sample_categorical(&mut self, probs: &[f32]) -> usize {
        let r = self.rng.next_f32();
        let mut acc = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            acc += p;
            if r < acc {
                return i;
            }
        }
        probs.len() - 1 // guard against floating-point shortfall
    }
}

impl Net {
    // --- Parameter lookup (forward) ---

    /// Returns a weight matrix as `(data, rows, cols)`, row-major with
    /// `rows` = output units and `cols` = input units.
    fn weight(&self, p: Param) -> (&[f32], usize, usize) {
        match p {
            Param::W1 => (&self.w1, NN_HID, NN_IN),
            Param::W2 => (&self.w2, NN_OUT, NN_HID),
            _ => panic!("not a weight id"),
        }
    }

    fn bias(&self, p: Param) -> &[f32] {
        match p {
            Param::B1 => &self.b1,
            Param::B2 => &self.b2,
            _ => panic!("not a bias id"),
        }
    }

    /// Returns the disjoint borrows DLinear needs in one call: the weight
    /// matrix (read) plus its weight and bias gradient buffers (written), with
    /// the matrix shape. Bundling them sidesteps the "two borrows of the same
    /// object" problem a separate-getter approach would hit.
    fn dlinear_refs(
        &mut self,
        w: Param,
        b: Param,
    ) -> (&[f32], &mut [f32], &mut [f32], usize, usize) {
        match (w, b) {
            (Param::W1, Param::B1) => (&self.w1, &mut self.g_w1, &mut self.g_b1, NN_HID, NN_IN),
            (Param::W2, Param::B2) => (&self.w2, &mut self.g_w2, &mut self.g_b2, NN_OUT, NN_HID),
            _ => panic!("bad dlinear params"),
        }
    }
}

// === --- VM ----------------------------------------------------------- ===

/// An index into the VM's tensor registry; the stack and tape carry these
/// instead of pointers. Replaces the C `struct nn_val` handle.
#[derive(Clone, Copy)]
struct TensorId(usize);

/// Execution state for one program run. The VM borrows only what it executes
/// against — the `prog` to run (shared) and the `net` it reads/writes (mut) —
/// and owns its scratch (`tensors`/`stack`/`tape`) plus the run I/O. Because
/// `prog` and `net` are distinct fields of [`HermesNn`], the VM can hold both
/// borrows at once without a clone.
struct Vm<'a> {
    prog: &'a Program,      // the instruction stream to execute
    net: &'a mut Net,       // params (read) and grad accumulators (written)
    tensors: Vec<Vec<f32>>, // registry; replaces the C float arena
    stack: Vec<TensorId>,   // operand stack
    tape: Vec<TensorId>,    // saved forward activations (LIFO)

    input: [f32; NN_IN],
    action: usize, // chosen action, for the REINFORCE seed
    reward: f32,   // scalar reward signal

    logp: f32,            // log pi(action), set by DSoftmaxReinforce
    probs: [f32; NN_OUT], // softmax output, set by Softmax
}

impl<'a> Vm<'a> {
    fn new(prog: &'a Program, net: &'a mut Net, input: [f32; NN_IN]) -> Vm<'a> {
        Vm {
            prog,
            net,
            tensors: Vec::new(),
            stack: Vec::new(),
            tape: Vec::new(),
            input,
            action: 0,
            reward: 0.0,
            logp: 0.0,
            probs: [0.0; NN_OUT],
        }
    }

    // --- registry / stack / tape helpers ---

    fn alloc(&mut self, data: Vec<f32>) -> TensorId {
        let id = TensorId(self.tensors.len());
        self.tensors.push(data);
        id
    }

    fn val(&self, id: TensorId) -> &[f32] {
        &self.tensors[id.0]
    }

    fn push(&mut self, id: TensorId) {
        self.stack.push(id);
    }

    fn pop(&mut self) -> TensorId {
        self.stack.pop().expect("operand stack underflow")
    }

    fn tape_push(&mut self, id: TensorId) {
        self.tape.push(id);
    }

    fn tape_pop(&mut self) -> TensorId {
        self.tape.pop().expect("tape underflow")
    }

    fn run(&mut self) {
        // Copy the program reference (references are `Copy`) so iterating it
        // doesn't tie up a borrow of `self` the ops need to mutate.
        let prog = self.prog;
        for instr in &prog.instr {
            match instr.op {
                Op::Input => self.op_input(),
                Op::Linear => self.op_linear(*instr),
                Op::Tanh => self.op_tanh(),
                Op::Softmax => self.op_softmax(),
                Op::DSoftmaxReinforce => self.op_dsoftmax_reinforce(),
                Op::DTanh => self.op_dtanh(),
                Op::DLinear => self.op_dlinear(*instr),
            }
        }
    }

    // --- forward ops ---

    fn op_input(&mut self) {
        let id = self.alloc(self.input.to_vec());
        self.push(id);
    }

    // pop x; save x for the backward partner; push W*x + b.
    fn op_linear(&mut self, instr: Instr) {
        let x_id = self.pop();
        self.tape_push(x_id);
        let x = self.val(x_id);

        let (w, rows, cols) = self.net.weight(instr.a);
        let b = self.net.bias(instr.b);
        assert_eq!(x.len(), cols, "linear: input length mismatch");

        let mut out = vec![0.0f32; rows];
        for o in 0..rows {
            let mut acc = b[o];
            for i in 0..cols {
                acc += w[o * cols + i] * x[i];
            }
            out[o] = acc;
        }
        let id = self.alloc(out);
        self.push(id);
    }

    // pop z; push tanh(z); save the output for DTanh.
    fn op_tanh(&mut self) {
        let z_id = self.pop();
        let out: Vec<f32> = self.val(z_id).iter().map(|v| v.tanh()).collect();
        let id = self.alloc(out);
        self.push(id);
        self.tape_push(id);
    }

    // pop logits; push softmax probs; save probs for DSoftmax; export probs.
    fn op_softmax(&mut self) {
        let z_id = self.pop();
        let z = self.val(z_id);

        let mut max = z[0];
        for &v in &z[1..] {
            if v > max {
                max = v;
            }
        }
        let sum: f32 = z.iter().map(|v| (v - max).exp()).sum();
        let out: Vec<f32> = z.iter().map(|v| (v - max).exp() / sum).collect();

        let n = out.len().min(NN_OUT);
        self.probs[..n].copy_from_slice(&out[..n]);
        let id = self.alloc(out);
        self.push(id);
        self.tape_push(id);
    }

    // --- backward ops ---

    // Seed the gradient on the output logits for REINFORCE:
    //   dL/dz = (probs - onehot(action)) * reward,  L = -log pi(action) * reward.
    // Also records logp = log pi(action).
    fn op_dsoftmax_reinforce(&mut self) {
        let p_id = self.tape_pop();
        let p = self.val(p_id);
        let a = self.action;
        assert!(a < p.len(), "action out of range");

        let logp = p[a].ln();
        let reward = self.reward;
        let g: Vec<f32> = p
            .iter()
            .enumerate()
            .map(|(i, &pi)| {
                let onehot = if i == a { 1.0 } else { 0.0 };
                (pi - onehot) * reward
            })
            .collect();
        self.logp = logp;
        let id = self.alloc(g);
        self.push(id);
    }

    // pop grad g (len rows); pop saved input x (len cols); accumulate
    //   g_W[o,i] += g[o] * x[i],  g_b[o] += g[o];  push grad wrt input W^T * g.
    fn op_dlinear(&mut self, instr: Instr) {
        let g_id = self.pop();
        let x_id = self.tape_pop();

        let gx = {
            // Index the `tensors` field directly (not via `self.val`, which
            // borrows all of `self`) so these shared borrows are field-disjoint
            // from the `&mut self.net` taken by `dlinear_refs`.
            let g = &self.tensors[g_id.0];
            let x = &self.tensors[x_id.0];
            let (w, gw, gb, rows, cols) = self.net.dlinear_refs(instr.a, instr.b);
            assert_eq!(g.len(), rows, "dlinear: grad length mismatch");
            assert_eq!(x.len(), cols, "dlinear: input length mismatch");

            for o in 0..rows {
                gb[o] += g[o];
                for i in 0..cols {
                    gw[o * cols + i] += g[o] * x[i];
                }
            }

            let mut gx = vec![0.0f32; cols];
            for i in 0..cols {
                let mut acc = 0.0;
                for o in 0..rows {
                    acc += w[o * cols + i] * g[o];
                }
                gx[i] = acc;
            }
            gx
        };
        let id = self.alloc(gx);
        self.push(id);
    }

    // pop grad g; pop saved tanh output h; push g * (1 - h^2).
    fn op_dtanh(&mut self) {
        let g_id = self.pop();
        let h_id = self.tape_pop();
        let g = self.val(g_id);
        let h = self.val(h_id);
        assert_eq!(g.len(), h.len(), "dtanh: length mismatch");

        let out: Vec<f32> = g
            .iter()
            .zip(h.iter())
            .map(|(&gi, &hi)| gi * (1.0 - hi * hi))
            .collect();
        let id = self.alloc(out);
        self.push(id);
    }
}

// === --- Tests -------------------------------------------------------- ===

#[cfg(test)]
mod tests {
    use super::*;

    // Forward-only loss L = -log pi(action) * reward, used for finite diffs.
    fn loss(nn: &mut HermesNn, pos: f32, speed: f32, action: Action, reward: f32) -> f32 {
        let probs = nn.forward_probs(pos, speed);
        -(probs[action as usize].ln()) * reward
    }

    /// Finite-difference check of the analytic REINFORCE gradient.
    #[test]
    fn grad_check() {
        let mut nn = HermesNn::new(42);
        let (pos, speed) = (0.3f32, -0.2f32);
        let action = Action::Left;
        let reward = 1.5f32;

        // Analytic gradients.
        nn.zero_grad();
        nn.accumulate(pos, speed, action, reward);
        let g_w1 = nn.net.g_w1;
        let g_b1 = nn.net.g_b1;
        let g_w2 = nn.net.g_w2;
        let g_b2 = nn.net.g_b2;

        let eps = 1e-3f32;
        let mut max_err = 0.0f32;

        // A tiny harness over each parameter block: perturb element k by +/-eps,
        // central-difference the loss, compare to the analytic grad.
        macro_rules! check_block {
            ($field:ident, $grad:expr) => {
                for k in 0..nn.net.$field.len() {
                    let orig = nn.net.$field[k];
                    nn.net.$field[k] = orig + eps;
                    let lp = loss(&mut nn, pos, speed, action, reward);
                    nn.net.$field[k] = orig - eps;
                    let lm = loss(&mut nn, pos, speed, action, reward);
                    nn.net.$field[k] = orig;
                    let num = (lp - lm) / (2.0 * eps);
                    max_err = max_err.max((num - $grad[k]).abs());
                }
            };
        }

        check_block!(w1, g_w1);
        check_block!(b1, g_b1);
        check_block!(w2, g_w2);
        check_block!(b2, g_b2);

        assert!(max_err < 1e-3, "max |analytic - numeric| = {max_err}");
    }

    #[test]
    fn probs_are_distribution() {
        let mut nn = HermesNn::new(7);
        let (_, probs) = nn.act(0.1, 0.5);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "probs sum = {sum}");
        for &p in &probs {
            assert!(p > 0.0, "prob not positive: {p}");
        }
    }

    #[test]
    fn sgd_increases_chosen_prob() {
        let mut nn = HermesNn::new(123);
        let (pos, speed) = (0.2f32, -0.4f32);
        let action = Action::Right;

        let before = nn.forward_probs(pos, speed)[action as usize];
        for _ in 0..5 {
            nn.zero_grad();
            nn.accumulate(pos, speed, action, 1.0);
            nn.sgd_step(0.1);
        }
        let after = nn.forward_probs(pos, speed)[action as usize];
        assert!(after > before, "prob did not rise: {before} -> {after}");
    }

    #[test]
    fn determinism() {
        let mut a = HermesNn::new(99);
        let mut b = HermesNn::new(99);
        assert_eq!(a.net.w1, b.net.w1);
        assert_eq!(a.net.w2, b.net.w2);
        let (act_a, _) = a.act(0.25, 0.0);
        let (act_b, _) = b.act(0.25, 0.0);
        assert_eq!(act_a, act_b);
    }
}
