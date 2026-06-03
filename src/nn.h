#ifndef HERMES_NN_H
#define HERMES_NN_H

#include "policy.h"

// === --- Topology ----------------------------------------------------- ===
//

// Observation dimensions fed to the network: position and speed.
#define HERMES_NN_IN 2
// Hidden units in the single tanh layer.
#define HERMES_NN_HID 16
// Output logits, one per action: none / left / right.
#define HERMES_NN_OUT 3
// Largest activation vector the VM stack must hold (max layer width).
#define HERMES_NN_MAX_DIM HERMES_NN_HID

// === --- Stack VM program model --------------------------------------- ===
//

// Opcodes for the stack VM. Forward ops push results on the value stack and
// record their output on the activation tape; backward ops consume gradients
// off the stack, read taped activations, and accumulate into the g_* buffers.
enum nn_op {
        // Forward.
        NN_OP_INPUT,   // push the observation vector (len IN)
        NN_OP_LINEAR,  // pop x, push W*x + b   (a=w_id, b=b_id, c=out_len)
        NN_OP_TANH,    // pop z, push tanh(z) elementwise
        NN_OP_SOFTMAX, // pop logits, push softmax probs; fills probs_out
        // Backward (training program only).
        NN_OP_DSOFTMAX_REINFORCE, // seed: (probs - onehot(action)) * advantage
        NN_OP_DTANH,              // grad *= 1 - tanh(z)^2  (uses taped output)
        NN_OP_DLINEAR,            // accumulate g_W, g_b; push W^T * grad
};

// Identifies a parameter block (weight matrix or bias vector) inside
// struct hermes_nn, so a LINEAR/DLINEAR instruction can reference it by id.
enum nn_param {
        NN_PARAM_W1,
        NN_PARAM_B1,
        NN_PARAM_W2,
        NN_PARAM_B2,
};

// A single VM instruction. The three integer operands are interpreted per
// opcode (see enum nn_op); unused operands are 0.
struct nn_instr {
        int op;  // enum nn_op
        int a;   // typically a weight param id (enum nn_param)
        int b;   // typically a bias param id (enum nn_param)
        int c;   // typically an output length
};

// Upper bound on instructions in a compiled program (forward + backward of the
// fixed two-layer topology fits comfortably).
#define HERMES_NN_PROG_MAX 16

// A compiled instruction stream produced by hermes_nn_compile.
struct nn_program {
        struct nn_instr instr[HERMES_NN_PROG_MAX];
        int             len;
};

// === --- Network ------------------------------------------------------ ===
//

// A fixed-topology MLP policy. All parameters and gradient accumulators are
// statically sized struct members (no heap). Weight matrices are row-major:
// w1[h * IN + i] is the weight from input i to hidden unit h.
struct hermes_nn {
        // Parameters.
        double w1[HERMES_NN_HID * HERMES_NN_IN];
        double b1[HERMES_NN_HID];
        double w2[HERMES_NN_OUT * HERMES_NN_HID];
        double b2[HERMES_NN_OUT];
        // Gradient accumulators, same shapes as the parameters above.
        double g_w1[HERMES_NN_HID * HERMES_NN_IN];
        double g_b1[HERMES_NN_HID];
        double g_w2[HERMES_NN_OUT * HERMES_NN_HID];
        double g_b2[HERMES_NN_OUT];
        // Compiled stack-VM programs (filled by hermes_nn_compile).
        struct nn_program forward;  // logits + softmax only
        struct nn_program train;    // forward followed by backward
};

// === --- API ---------------------------------------------------------- ===
//

// Initializes parameters with small random values and zeroes the gradients.
// seed makes the initialization reproducible.
void hermes_nn_init( struct hermes_nn *nn, unsigned seed );

// Builds the forward and training stack-VM programs from the fixed topology.
// Idempotent; call once after init before any forward/training pass.
void hermes_nn_compile( struct hermes_nn *nn );

// Runs the forward program for the observation (pos, speed), writes the action
// probabilities into probs_out, and returns an action sampled from that
// categorical distribution.
enum hermes_action hermes_nn_act( struct hermes_nn *nn, double pos, double speed,
                                  double probs_out[HERMES_NN_OUT] );

// Runs the training program for one REINFORCE sample, accumulating the policy
// gradient of -log pi(action | obs) * advantage into the g_* buffers. Returns
// log pi(action | obs). Does not modify parameters; call hermes_nn_sgd_step to
// apply the accumulated gradients.
double hermes_nn_accumulate( struct hermes_nn *nn, double pos, double speed,
                             enum hermes_action action, double advantage );

// Zeroes all gradient accumulators.
void hermes_nn_zero_grad( struct hermes_nn *nn );

// Applies a single SGD update p -= lr * g across all parameters.
void hermes_nn_sgd_step( struct hermes_nn *nn, double lr );

#endif
