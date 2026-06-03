#include "nn.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// === --- VM value, tape, and state ------------------------------------ ===
//

// A single activation vector flowing through the VM (an operand on the value
// stack, or a saved activation on the tape).
struct nn_val {
        double v[HERMES_NN_MAX_DIM];
        int    len;
};

// Depth bounds for the two LIFO buffers. The fixed topology threads one live
// value at a time, but a small margin keeps the VM robust to extra ops.
#define NN_STACK_MAX 4
// One tape slot per forward op that saves an activation for its backward
// partner (LINEAR, TANH, SOFTMAX); margin included.
#define NN_TAPE_MAX 8

// Execution state for one program run.
struct nn_vm {
        struct hermes_nn *nn;
        struct nn_val     stack[NN_STACK_MAX];  // operand stack
        int               sp;                   // operand stack depth
        struct nn_val     tape[NN_TAPE_MAX];    // saved forward activations
        int               tp;                   // tape depth

        const double *input;     // observation, length HERMES_NN_IN
        int           action;    // chosen action, for the REINFORCE seed
        double        advantage; // scalar reward signal

        double logp;                     // log pi(action), set by DSOFTMAX
        double probs[HERMES_NN_OUT];     // softmax output, set by SOFTMAX
};

// === --- Stack / tape helpers ----------------------------------------- ===
//

static struct nn_val *
nn_push( struct nn_vm *vm, int len )
{
        assert( vm->sp < NN_STACK_MAX );
        assert( len > 0 && len <= HERMES_NN_MAX_DIM );
        struct nn_val *val = &vm->stack[vm->sp++];
        val->len           = len;
        return val;
}

static struct nn_val *
nn_pop( struct nn_vm *vm )
{
        assert( vm->sp > 0 );
        return &vm->stack[--vm->sp];
}

static void
nn_tape_push( struct nn_vm *vm, const struct nn_val *val )
{
        assert( vm->tp < NN_TAPE_MAX );
        vm->tape[vm->tp++] = *val;
}

static struct nn_val *
nn_tape_pop( struct nn_vm *vm )
{
        assert( vm->tp > 0 );
        return &vm->tape[--vm->tp];
}

// === --- Parameter lookup --------------------------------------------- ===
//

// A weight matrix and its gradient buffer, with its row-major shape
// (rows = output units, cols = input units).
struct nn_weight {
        double *w;
        double *g;
        int     rows;
        int     cols;
};

static struct nn_weight
nn_weight_of( struct hermes_nn *nn, int w_id )
{
        switch ( w_id ) {
        case NN_PARAM_W1:
                return ( struct nn_weight ){ nn->w1, nn->g_w1, HERMES_NN_HID,
                                             HERMES_NN_IN };
        case NN_PARAM_W2:
                return ( struct nn_weight ){ nn->w2, nn->g_w2, HERMES_NN_OUT,
                                             HERMES_NN_HID };
        }
        assert( 0 && "bad weight id" );
        return ( struct nn_weight ){ 0 };
}

static double *
nn_bias_of( struct hermes_nn *nn, int b_id )
{
        switch ( b_id ) {
        case NN_PARAM_B1:
                return nn->b1;
        case NN_PARAM_B2:
                return nn->b2;
        }
        assert( 0 && "bad bias id" );
        return NULL;
}

static double *
nn_bias_grad_of( struct hermes_nn *nn, int b_id )
{
        switch ( b_id ) {
        case NN_PARAM_B1:
                return nn->g_b1;
        case NN_PARAM_B2:
                return nn->g_b2;
        }
        assert( 0 && "bad bias id" );
        return NULL;
}

// === --- Forward ops -------------------------------------------------- ===
//

static void
op_input( struct nn_vm *vm )
{
        struct nn_val *out = nn_push( vm, HERMES_NN_IN );
        for ( int i = 0; i < HERMES_NN_IN; i++ ) out->v[i] = vm->input[i];
}

// pop x; save x for the backward partner; push W*x + b.
static void
op_linear( struct nn_vm *vm, const struct nn_instr *in )
{
        struct nn_weight wt = nn_weight_of( vm->nn, in->a );
        const double    *b  = nn_bias_of( vm->nn, in->b );

        struct nn_val x = *nn_pop( vm );
        assert( x.len == wt.cols );
        nn_tape_push( vm, &x );

        struct nn_val *out = nn_push( vm, wt.rows );
        for ( int o = 0; o < wt.rows; o++ ) {
                double acc = b[o];
                for ( int i = 0; i < wt.cols; i++ )
                        acc += wt.w[o * wt.cols + i] * x.v[i];
                out->v[o] = acc;
        }
}

// pop z; push tanh(z); save the output for DTANH.
static void
op_tanh( struct nn_vm *vm )
{
        struct nn_val z   = *nn_pop( vm );
        struct nn_val *out = nn_push( vm, z.len );
        for ( int i = 0; i < z.len; i++ ) out->v[i] = tanh( z.v[i] );
        nn_tape_push( vm, out );
}

// pop logits; push softmax probs; save probs for DSOFTMAX; export probs.
static void
op_softmax( struct nn_vm *vm )
{
        struct nn_val z = *nn_pop( vm );

        double max = z.v[0];
        for ( int i = 1; i < z.len; i++ )
                if ( z.v[i] > max ) max = z.v[i];

        double sum = 0.0;
        for ( int i = 0; i < z.len; i++ ) sum += exp( z.v[i] - max );

        struct nn_val *out = nn_push( vm, z.len );
        for ( int i = 0; i < z.len; i++ )
                out->v[i] = exp( z.v[i] - max ) / sum;

        nn_tape_push( vm, out );
        for ( int i = 0; i < z.len; i++ ) vm->probs[i] = out->v[i];
}

// === --- Backward ops ------------------------------------------------- ===
//

// Seed the gradient on the output logits for REINFORCE:
//   dL/dz = (probs - onehot(action)) * advantage,   L = -log pi(action) * adv.
// Also records logp = log pi(action).
static void
op_dsoftmax_reinforce( struct nn_vm *vm )
{
        struct nn_val p = *nn_tape_pop( vm );
        assert( vm->action >= 0 && vm->action < p.len );

        vm->logp = log( p.v[vm->action] );

        struct nn_val *g = nn_push( vm, p.len );
        for ( int i = 0; i < p.len; i++ ) {
                double onehot = ( i == vm->action ) ? 1.0 : 0.0;
                g->v[i]       = ( p.v[i] - onehot ) * vm->advantage;
        }
}

// pop grad g (len rows); pop saved input x (len cols); accumulate
//   g_W[o,i] += g[o] * x[i],  g_b[o] += g[o];  push grad wrt input W^T * g.
static void
op_dlinear( struct nn_vm *vm, const struct nn_instr *in )
{
        struct nn_weight wt = nn_weight_of( vm->nn, in->a );
        double          *gb = nn_bias_grad_of( vm->nn, in->b );

        struct nn_val g = *nn_pop( vm );
        struct nn_val x = *nn_tape_pop( vm );
        assert( g.len == wt.rows && x.len == wt.cols );

        for ( int o = 0; o < wt.rows; o++ ) {
                gb[o] += g.v[o];
                for ( int i = 0; i < wt.cols; i++ )
                        wt.g[o * wt.cols + i] += g.v[o] * x.v[i];
        }

        struct nn_val *gx = nn_push( vm, wt.cols );
        for ( int i = 0; i < wt.cols; i++ ) {
                double acc = 0.0;
                for ( int o = 0; o < wt.rows; o++ )
                        acc += wt.w[o * wt.cols + i] * g.v[o];
                gx->v[i] = acc;
        }
}

// pop grad g; pop saved tanh output h; push g * (1 - h^2).
static void
op_dtanh( struct nn_vm *vm )
{
        struct nn_val g = *nn_pop( vm );
        struct nn_val h = *nn_tape_pop( vm );
        assert( g.len == h.len );

        struct nn_val *out = nn_push( vm, g.len );
        for ( int i = 0; i < g.len; i++ )
                out->v[i] = g.v[i] * ( 1.0 - h.v[i] * h.v[i] );
}

// === --- VM engine ---------------------------------------------------- ===
//

static void
nn_vm_run( struct nn_vm *vm, const struct nn_program *prog )
{
        for ( int pc = 0; pc < prog->len; pc++ ) {
                const struct nn_instr *in = &prog->instr[pc];
                switch ( in->op ) {
                case NN_OP_INPUT:
                        op_input( vm );
                        break;
                case NN_OP_LINEAR:
                        op_linear( vm, in );
                        break;
                case NN_OP_TANH:
                        op_tanh( vm );
                        break;
                case NN_OP_SOFTMAX:
                        op_softmax( vm );
                        break;
                case NN_OP_DSOFTMAX_REINFORCE:
                        op_dsoftmax_reinforce( vm );
                        break;
                case NN_OP_DTANH:
                        op_dtanh( vm );
                        break;
                case NN_OP_DLINEAR:
                        op_dlinear( vm, in );
                        break;
                default:
                        assert( 0 && "bad opcode" );
                }
        }
}

// === --- Compilation -------------------------------------------------- ===
//

static void
nn_emit( struct nn_program *prog, int op, int a, int b, int c )
{
        assert( prog->len < HERMES_NN_PROG_MAX );
        prog->instr[prog->len++] = ( struct nn_instr ){ op, a, b, c };
}

void
hermes_nn_compile( struct hermes_nn *nn )
{
        struct nn_program *fwd = &nn->forward;
        fwd->len               = 0;
        nn_emit( fwd, NN_OP_INPUT, 0, 0, 0 );
        nn_emit( fwd, NN_OP_LINEAR, NN_PARAM_W1, NN_PARAM_B1, HERMES_NN_HID );
        nn_emit( fwd, NN_OP_TANH, 0, 0, 0 );
        nn_emit( fwd, NN_OP_LINEAR, NN_PARAM_W2, NN_PARAM_B2, HERMES_NN_OUT );
        nn_emit( fwd, NN_OP_SOFTMAX, 0, 0, 0 );

        // The training program is the forward pass followed by its reverse.
        struct nn_program *trn = &nn->train;
        trn->len               = 0;
        for ( int i = 0; i < fwd->len; i++ ) trn->instr[trn->len++] = fwd->instr[i];
        nn_emit( trn, NN_OP_DSOFTMAX_REINFORCE, 0, 0, 0 );
        nn_emit( trn, NN_OP_DLINEAR, NN_PARAM_W2, NN_PARAM_B2, 0 );
        nn_emit( trn, NN_OP_DTANH, 0, 0, 0 );
        nn_emit( trn, NN_OP_DLINEAR, NN_PARAM_W1, NN_PARAM_B1, 0 );
}

// === --- Init / optimizer --------------------------------------------- ===
//

// Half-width of the uniform range for initial weights: w ~ U(-SCALE, SCALE).
#define NN_INIT_SCALE 0.1

static double
nn_uniform( void )
{
        return ( (double)rand( ) / ( (double)RAND_MAX + 1.0 ) ) * 2.0 - 1.0;
}

static void
nn_init_weights( double *w, int n )
{
        for ( int i = 0; i < n; i++ ) w[i] = nn_uniform( ) * NN_INIT_SCALE;
}

void
hermes_nn_init( struct hermes_nn *nn, unsigned seed )
{
        srand( seed );
        nn_init_weights( nn->w1, HERMES_NN_HID * HERMES_NN_IN );
        nn_init_weights( nn->w2, HERMES_NN_OUT * HERMES_NN_HID );
        memset( nn->b1, 0, sizeof( nn->b1 ) );
        memset( nn->b2, 0, sizeof( nn->b2 ) );
        hermes_nn_zero_grad( nn );
}

void
hermes_nn_zero_grad( struct hermes_nn *nn )
{
        memset( nn->g_w1, 0, sizeof( nn->g_w1 ) );
        memset( nn->g_b1, 0, sizeof( nn->g_b1 ) );
        memset( nn->g_w2, 0, sizeof( nn->g_w2 ) );
        memset( nn->g_b2, 0, sizeof( nn->g_b2 ) );
}

static void
nn_sgd_array( double *w, const double *g, int n, double lr )
{
        for ( int i = 0; i < n; i++ ) w[i] -= lr * g[i];
}

void
hermes_nn_sgd_step( struct hermes_nn *nn, double lr )
{
        nn_sgd_array( nn->w1, nn->g_w1, HERMES_NN_HID * HERMES_NN_IN, lr );
        nn_sgd_array( nn->b1, nn->g_b1, HERMES_NN_HID, lr );
        nn_sgd_array( nn->w2, nn->g_w2, HERMES_NN_OUT * HERMES_NN_HID, lr );
        nn_sgd_array( nn->b2, nn->g_b2, HERMES_NN_OUT, lr );
}

// === --- Forward / training entry points ------------------------------ ===
//

static int
nn_sample_categorical( const double *probs, int n )
{
        double r   = (double)rand( ) / ( (double)RAND_MAX + 1.0 );
        double acc = 0.0;
        for ( int i = 0; i < n; i++ ) {
                acc += probs[i];
                if ( r < acc ) return i;
        }
        return n - 1;  // guard against floating-point shortfall
}

enum hermes_action
hermes_nn_act( struct hermes_nn *nn, double pos, double speed,
               double probs_out[HERMES_NN_OUT] )
{
        double      obs[HERMES_NN_IN] = { pos, speed };
        struct nn_vm vm               = { 0 };
        vm.nn                         = nn;
        vm.input                      = obs;

        nn_vm_run( &vm, &nn->forward );

        for ( int i = 0; i < HERMES_NN_OUT; i++ ) probs_out[i] = vm.probs[i];
        return ( enum hermes_action )nn_sample_categorical( vm.probs,
                                                            HERMES_NN_OUT );
}

double
hermes_nn_accumulate( struct hermes_nn *nn, double pos, double speed,
                      enum hermes_action action, double advantage )
{
        double      obs[HERMES_NN_IN] = { pos, speed };
        struct nn_vm vm               = { 0 };
        vm.nn                         = nn;
        vm.input                      = obs;
        vm.action                     = (int)action;
        vm.advantage                  = advantage;

        nn_vm_run( &vm, &nn->train );
        return vm.logp;
}
