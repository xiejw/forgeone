#include "nn.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// === --- VM value, tape, and state ------------------------------------ ===
//

// A handle to an activation living in the VM's scratch arena: where it starts
// and how many elements it spans. Carries no data, so stack/tape ops copy only
// this descriptor, never the underlying array.
struct nn_val {
        float *v;    // points into vm->arena
        int    len;  // number of elements
};

// Depth bounds for the two LIFO buffers. The fixed topology threads one live
// value at a time, but a small margin keeps the VM robust to extra ops.
#define NN_STACK_MAX 4
// One tape slot per forward op that saves an activation for its backward
// partner (LINEAR, TANH, SOFTMAX); margin included.
#define NN_TAPE_MAX 8
// Floats the scratch arena must hold for one run. Outputs are never reclaimed
// mid-run (forward activations stay live for the backward pass), so this is the
// sum of every op's output length. The train program allocates ~77 floats
// (INPUT 2 + LINEAR 16 + TANH 16 + LINEAR 3 + SOFTMAX 3 + DSOFTMAX 3 +
// DLINEAR 16 + DTANH 16 + DLINEAR 2); rounded up with margin.
#define NN_ARENA_MAX 128

// Execution state for one program run.
struct nn_vm {
        struct hermes_nn *nn;

        float arena[NN_ARENA_MAX];  // activation storage
        int   arena_used;           // floats handed out so far

        struct nn_val stack[NN_STACK_MAX];  // operand stack (handles)
        int           sp;                   // operand stack depth

        struct nn_val tape[NN_TAPE_MAX];  // saved forward activations
        int           tp;                 // tape depth

        // Set when a bounds/shape check fails (see NN_REQUIRE). assert()
        // catches violations immediately in debug builds; in release builds
        // (NDEBUG, asserts compiled out) the bit instead lets the program
        // finish writing into the safe scratch sink and then aborts in
        // nn_vm_run, rather than letting a bad index silently corrupt state.
        int dirty;
        // The sink is a throwaway scratch value handed back by the stack/tape
        // helpers once a check has failed. With asserts compiled out the run
        // keeps going, so the failing op still reads and writes *something*;
        // routing it here — instead of an out-of-range stack/arena slot — keeps
        // every access inside sink_buf, so the program reaches nn_vm_run's
        // abort without an out-of-bounds touch. sink_buf is sized to the
        // largest tensor (HERMES_NN_MAX_DIM) so any single op's write fits.
        float         sink_buf[HERMES_NN_MAX_DIM];
        struct nn_val sink;  // handle pointing at sink_buf, armed in nn_vm_run

        const float *input;   // observation, length HERMES_NN_IN
        int          action;  // chosen action, for the REINFORCE seed
        float        reward;  // scalar reward signal

        float logp;                  // log pi(action), set by DSOFTMAX
        float probs[HERMES_NN_OUT];  // softmax output, set by SOFTMAX
};

// === --- Stack / tape helpers ----------------------------------------- ===
//

// Requires a VM invariant to hold. In debug builds a violation aborts
// immediately via assert(); in release builds (NDEBUG, asserts compiled out) it
// instead sets vm->dirty so the helpers fall back to the safe scratch sink and
// nn_vm_run aborts once the program finishes (see struct nn_vm::dirty).
#define NN_REQUIRE( vm, cond )             \
        do {                               \
                if ( !( cond ) ) {         \
                        ( vm )->dirty = 1; \
                        assert( cond );    \
                }                          \
        } while ( 0 )

// Push a fresh value: bump-allocate len floats from the arena and return a
// handle to them. The new region never aliases any prior value, so ops can read
// their input and write their output without copying the input out first. On a
// failed check the value goes to the sink so the write stays in bounds.
static struct nn_val *
nn_push( struct nn_vm *vm, int len )
{
        NN_REQUIRE( vm, vm->sp < NN_STACK_MAX );
        NN_REQUIRE( vm, len > 0 && len <= HERMES_NN_MAX_DIM );
        NN_REQUIRE( vm, vm->arena_used + len <= NN_ARENA_MAX );
        if ( vm->dirty ) return &vm->sink;
        struct nn_val *val = &vm->stack[vm->sp++];
        val->v             = &vm->arena[vm->arena_used];
        val->len           = len;
        vm->arena_used += len;
        return val;
}

static struct nn_val *
nn_pop( struct nn_vm *vm )
{
        NN_REQUIRE( vm, vm->sp > 0 );
        if ( vm->dirty ) return &vm->sink;
        return &vm->stack[--vm->sp];
}

static void
nn_tape_push( struct nn_vm *vm, const struct nn_val *val )
{
        NN_REQUIRE( vm, vm->tp < NN_TAPE_MAX );
        if ( vm->dirty ) return;
        vm->tape[vm->tp++] = *val;
}

static struct nn_val *
nn_tape_pop( struct nn_vm *vm )
{
        NN_REQUIRE( vm, vm->tp > 0 );
        if ( vm->dirty ) return &vm->sink;
        return &vm->tape[--vm->tp];
}

// === --- Parameter lookup --------------------------------------------- ===
//

// A weight matrix and its gradient buffer, with its row-major shape
// (rows = output units, cols = input units).
struct nn_weight {
        float *w;
        float *g;
        int    rows;
        int    cols;
};

static struct nn_weight
nn_weight_of( struct nn_vm *vm, int w_id )
{
        struct hermes_nn *nn = vm->nn;
        switch ( w_id ) {
        case NN_PARAM_W1:
                return (struct nn_weight){ nn->w1, nn->g_w1, HERMES_NN_HID,
                                           HERMES_NN_IN };
        case NN_PARAM_W2:
                return (struct nn_weight){ nn->w2, nn->g_w2, HERMES_NN_OUT,
                                           HERMES_NN_HID };
        }
        NN_REQUIRE( vm, 0 && "bad weight id" );
        return (struct nn_weight){ 0 };
}

static float *
nn_bias_of( struct nn_vm *vm, int b_id )
{
        switch ( b_id ) {
        case NN_PARAM_B1:
                return vm->nn->b1;
        case NN_PARAM_B2:
                return vm->nn->b2;
        }
        NN_REQUIRE( vm, 0 && "bad bias id" );
        return vm->sink_buf;
}

static float *
nn_bias_grad_of( struct nn_vm *vm, int b_id )
{
        switch ( b_id ) {
        case NN_PARAM_B1:
                return vm->nn->g_b1;
        case NN_PARAM_B2:
                return vm->nn->g_b2;
        }
        NN_REQUIRE( vm, 0 && "bad bias id" );
        return vm->sink_buf;
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
        struct nn_weight wt = nn_weight_of( vm, in->a );
        const float     *b  = nn_bias_of( vm, in->b );

        struct nn_val x = *nn_pop( vm );
        NN_REQUIRE( vm, x.len == wt.cols );
        nn_tape_push( vm, &x );

        struct nn_val *out = nn_push( vm, wt.rows );
        for ( int o = 0; o < wt.rows; o++ ) {
                float acc = b[o];
                for ( int i = 0; i < wt.cols; i++ )
                        acc += wt.w[o * wt.cols + i] * x.v[i];
                out->v[o] = acc;
        }
}

// pop z; push tanh(z); save the output for DTANH.
static void
op_tanh( struct nn_vm *vm )
{
        struct nn_val  z   = *nn_pop( vm );
        struct nn_val *out = nn_push( vm, z.len );
        for ( int i = 0; i < z.len; i++ ) out->v[i] = tanhf( z.v[i] );
        nn_tape_push( vm, out );
}

// pop logits; push softmax probs; save probs for DSOFTMAX; export probs.
static void
op_softmax( struct nn_vm *vm )
{
        struct nn_val z = *nn_pop( vm );

        float max = z.v[0];
        for ( int i = 1; i < z.len; i++ )
                if ( z.v[i] > max ) max = z.v[i];

        float sum = 0.0f;
        for ( int i = 0; i < z.len; i++ ) sum += expf( z.v[i] - max );

        struct nn_val *out = nn_push( vm, z.len );
        for ( int i = 0; i < z.len; i++ )
                out->v[i] = expf( z.v[i] - max ) / sum;

        nn_tape_push( vm, out );
        for ( int i = 0; i < z.len; i++ ) vm->probs[i] = out->v[i];
}

// === --- Backward ops ------------------------------------------------- ===
//

// Seed the gradient on the output logits for REINFORCE:
//   dL/dz = (probs - onehot(action)) * reward,   L = -log pi(action) * reward.
// Also records logp = log pi(action).
static void
op_dsoftmax_reinforce( struct nn_vm *vm )
{
        struct nn_val p = *nn_tape_pop( vm );
        NN_REQUIRE( vm, vm->action >= 0 && vm->action < p.len );

        vm->logp = logf( p.v[vm->action] );

        struct nn_val *g = nn_push( vm, p.len );
        for ( int i = 0; i < p.len; i++ ) {
                float onehot = ( i == vm->action ) ? 1.0f : 0.0f;
                g->v[i]      = ( p.v[i] - onehot ) * vm->reward;
        }
}

// pop grad g (len rows); pop saved input x (len cols); accumulate
//   g_W[o,i] += g[o] * x[i],  g_b[o] += g[o];  push grad wrt input W^T * g.
static void
op_dlinear( struct nn_vm *vm, const struct nn_instr *in )
{
        struct nn_weight wt = nn_weight_of( vm, in->a );
        float           *gb = nn_bias_grad_of( vm, in->b );

        struct nn_val g = *nn_pop( vm );
        struct nn_val x = *nn_tape_pop( vm );
        NN_REQUIRE( vm, g.len == wt.rows && x.len == wt.cols );

        for ( int o = 0; o < wt.rows; o++ ) {
                gb[o] += g.v[o];
                for ( int i = 0; i < wt.cols; i++ )
                        wt.g[o * wt.cols + i] += g.v[o] * x.v[i];
        }

        struct nn_val *gx = nn_push( vm, wt.cols );
        for ( int i = 0; i < wt.cols; i++ ) {
                float acc = 0.0f;
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
        NN_REQUIRE( vm, g.len == h.len );

        struct nn_val *out = nn_push( vm, g.len );
        for ( int i = 0; i < g.len; i++ )
                out->v[i] = g.v[i] * ( 1.0f - h.v[i] * h.v[i] );
}

// === --- VM engine ---------------------------------------------------- ===
//

static void
nn_vm_run( struct nn_vm *vm, const struct nn_program *prog )
{
        // Arm the scratch sink so a failed check has somewhere safe to write.
        vm->sink.v   = vm->sink_buf;
        vm->sink.len = HERMES_NN_MAX_DIM;

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
                        NN_REQUIRE( vm, 0 && "bad opcode" );
                }
                // A violated invariant means the rest of the run would operate
                // on degraded state; stop before the next op reads it.
                if ( vm->dirty ) break;
        }

        // In debug builds an assert already aborted at the failing check. In
        // release builds this is the backstop: fail loud rather than return a
        // silently corrupt result.
        if ( vm->dirty ) {
                fprintf( stderr,
                         "hermes_nn: VM invariant violated; aborting\n" );
                abort( );
        }
}

// === --- Compilation -------------------------------------------------- ===
//

static void
nn_emit( struct nn_program *prog, int op, int a, int b, int c )
{
        assert( prog->len < HERMES_NN_PROG_MAX );
        prog->instr[prog->len++] = (struct nn_instr){ op, a, b, c };
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
        for ( int i = 0; i < fwd->len; i++ )
                trn->instr[trn->len++] = fwd->instr[i];
        nn_emit( trn, NN_OP_DSOFTMAX_REINFORCE, 0, 0, 0 );
        nn_emit( trn, NN_OP_DLINEAR, NN_PARAM_W2, NN_PARAM_B2, 0 );
        nn_emit( trn, NN_OP_DTANH, 0, 0, 0 );
        nn_emit( trn, NN_OP_DLINEAR, NN_PARAM_W1, NN_PARAM_B1, 0 );
}

// === --- Init / optimizer --------------------------------------------- ===
//

// Half-width of the uniform range for initial weights: w ~ U(-SCALE, SCALE).
#define NN_INIT_SCALE 0.1f

static float
nn_uniform( void )
{
        return ( (float)rand( ) / ( (float)RAND_MAX + 1.0f ) ) * 2.0f - 1.0f;
}

static void
nn_init_weights( float *w, int n )
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
nn_sgd_array( float *w, const float *g, int n, float lr )
{
        for ( int i = 0; i < n; i++ ) w[i] -= lr * g[i];
}

void
hermes_nn_sgd_step( struct hermes_nn *nn, float lr )
{
        nn_sgd_array( nn->w1, nn->g_w1, HERMES_NN_HID * HERMES_NN_IN, lr );
        nn_sgd_array( nn->b1, nn->g_b1, HERMES_NN_HID, lr );
        nn_sgd_array( nn->w2, nn->g_w2, HERMES_NN_OUT * HERMES_NN_HID, lr );
        nn_sgd_array( nn->b2, nn->g_b2, HERMES_NN_OUT, lr );
}

// === --- Forward / training entry points ------------------------------ ===
//

static int
nn_sample_categorical( const float *probs, int n )
{
        float r   = (float)rand( ) / ( (float)RAND_MAX + 1.0f );
        float acc = 0.0f;
        for ( int i = 0; i < n; i++ ) {
                acc += probs[i];
                if ( r < acc ) return i;
        }
        return n - 1;  // guard against floating-point shortfall
}

enum hermes_action
hermes_nn_act( struct hermes_nn *nn, float pos, float speed,
               float probs_out[HERMES_NN_OUT] )
{
        float        obs[HERMES_NN_IN] = { pos, speed };
        struct nn_vm vm                = { 0 };
        vm.nn                          = nn;
        vm.input                       = obs;

        nn_vm_run( &vm, &nn->forward );

        for ( int i = 0; i < HERMES_NN_OUT; i++ ) probs_out[i] = vm.probs[i];
        return (enum hermes_action)nn_sample_categorical( vm.probs,
                                                          HERMES_NN_OUT );
}

float
hermes_nn_accumulate( struct hermes_nn *nn, float pos, float speed,
                      enum hermes_action action, float reward )
{
        float        obs[HERMES_NN_IN] = { pos, speed };
        struct nn_vm vm                = { 0 };
        vm.nn                          = nn;
        vm.input                       = obs;
        vm.action                      = (int)action;
        vm.reward                      = reward;

        nn_vm_run( &vm, &nn->train );
        return vm.logp;
}
