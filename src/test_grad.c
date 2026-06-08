// Numerical-correctness tests for the LINEAR and SOFTMAX ops, forward and
// backward. White-box by design: it #includes nn.c so it can drive the static
// VM ops in isolation. Forward outputs are checked against a double-precision
// reference; backward gradients are checked against central finite differences
// of the same reference loss (the gradient-check gate from doc/DESIGN.md).

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nn.c"

// Forward outputs match a double reference to float rounding; finite-difference
// gradients carry truncation + float roundoff, so the backward gate is looser.
#define FWD_TOL 1e-4
#define BWD_TOL 1e-3
// Central-difference step, in double precision (small enough that truncation is
// far below the gate, large enough to stay clear of double roundoff).
#define FD_EPS 1e-4

// Track the worst |analytic - numeric| seen within a test.
#define UPD( maxerr, a, n )                                  \
        do {                                                 \
                double e_ = fabs( (double)( a ) - (double)( n ) ); \
                if ( e_ > ( maxerr ) ) ( maxerr ) = e_;      \
        } while ( 0 )

// === --- Helpers ------------------------------------------------------ ===
//

static double
rnd( double lo, double hi )
{
        return lo + ( hi - lo ) * ( (double)rand( ) / (double)RAND_MAX );
}

static void
randomize( double *a, int n, double lo, double hi )
{
        for ( int i = 0; i < n; i++ ) a[i] = rnd( lo, hi );
}

static int
report( const char *tag, double maxerr, double tol )
{
        if ( maxerr > tol ) {
                fprintf( stderr, "[%s] FAIL max_err=%.3g > %.0e\n", tag, maxerr,
                         tol );
                return 1;
        }
        printf( "[%s] max_err=%.3g (tol %.0e) OK\n", tag, maxerr, tol );
        return 0;
}

// Double-precision references (the "truth" the float ops are checked against).

static void
ref_linear( const double *W, const double *b, const double *x, int rows,
            int cols, double *out )
{
        for ( int o = 0; o < rows; o++ ) {
                double acc = b[o];
                for ( int i = 0; i < cols; i++ ) acc += W[o * cols + i] * x[i];
                out[o] = acc;
        }
}

static void
ref_softmax( const double *z, int n, double *p )
{
        double max = z[0];
        for ( int i = 1; i < n; i++ )
                if ( z[i] > max ) max = z[i];
        double sum = 0.0;
        for ( int i = 0; i < n; i++ ) sum += exp( z[i] - max );
        for ( int i = 0; i < n; i++ ) p[i] = exp( z[i] - max ) / sum;
}

// Scalar losses used for the finite-difference gradient checks.

// L = <g, W x + b>, whose gradients are exactly what DLINEAR computes:
// dL/dW = g x^T, dL/db = g, dL/dx = W^T g.
static double
loss_linear( const double *W, const double *b, const double *x, const double *g,
             int rows, int cols )
{
        double out[HERMES_NN_MAX_DIM];
        ref_linear( W, b, x, rows, cols, out );
        double L = 0.0;
        for ( int o = 0; o < rows; o++ ) L += g[o] * out[o];
        return L;
}

// L = -log(softmax(z)[action]) * reward, the REINFORCE seed DSOFTMAX produces.
static double
loss_softmax( const double *z, int n, int action, double reward )
{
        double p[HERMES_NN_MAX_DIM];
        ref_softmax( z, n, p );
        return -log( p[action] ) * reward;
}

// Drivers that run a single op in isolation through a fresh VM.

static void
arm( struct nn_vm *vm, struct hermes_nn *nn )
{
        memset( vm, 0, sizeof( *vm ) );
        vm->nn       = nn;
        vm->sink.v   = vm->sink_buf;
        vm->sink.len = HERMES_NN_MAX_DIM;
}

static void
drive_linear( struct hermes_nn *nn, int w_id, int b_id, int rows, int cols,
              const float *x, float *out )
{
        struct nn_vm vm;
        arm( &vm, nn );
        struct nn_val *xv = nn_push( &vm, cols );
        memcpy( xv->v, x, sizeof( float ) * (size_t)cols );

        struct nn_instr in = { NN_OP_LINEAR, w_id, b_id, rows };
        op_linear( &vm, &in );

        memcpy( out, nn_pop( &vm )->v, sizeof( float ) * (size_t)rows );
}

static void
drive_softmax( struct hermes_nn *nn, const float *z, int n, float *p )
{
        struct nn_vm vm;
        arm( &vm, nn );
        struct nn_val *zv = nn_push( &vm, n );
        memcpy( zv->v, z, sizeof( float ) * (size_t)n );

        op_softmax( &vm );

        memcpy( p, nn_pop( &vm )->v, sizeof( float ) * (size_t)n );
}

static void
drive_dlinear( struct hermes_nn *nn, int w_id, int b_id, int rows, int cols,
               const float *x, const float *g, float *gx )
{
        struct nn_vm vm;
        arm( &vm, nn );
        // x is the activation LINEAR saved on the tape; g is the upstream grad
        // on the operand stack.
        struct nn_val *xv = nn_push( &vm, cols );
        memcpy( xv->v, x, sizeof( float ) * (size_t)cols );
        nn_tape_push( &vm, xv );
        nn_pop( &vm );  // clear the operand stack; the tape keeps x

        struct nn_val *gv = nn_push( &vm, rows );
        memcpy( gv->v, g, sizeof( float ) * (size_t)rows );

        struct nn_instr in = { NN_OP_DLINEAR, w_id, b_id, 0 };
        op_dlinear( &vm, &in );

        memcpy( gx, nn_pop( &vm )->v, sizeof( float ) * (size_t)cols );
}

static void
drive_dsoftmax( struct hermes_nn *nn, const float *probs, int n, int action,
                float reward, float *dz, float *logp_out )
{
        struct nn_vm vm;
        arm( &vm, nn );
        vm.action = action;
        vm.reward = reward;
        // DSOFTMAX reads the taped probs (not the operand stack).
        struct nn_val *pv = nn_push( &vm, n );
        memcpy( pv->v, probs, sizeof( float ) * (size_t)n );
        nn_tape_push( &vm, pv );
        nn_pop( &vm );

        op_dsoftmax_reinforce( &vm );

        memcpy( dz, nn_pop( &vm )->v, sizeof( float ) * (size_t)n );
        *logp_out = vm.logp;
}

// === --- Tests -------------------------------------------------------- ===
//

// Forward: LINEAR output matches W x + b. Uses the W2 layer (rows OUT, cols
// HID) to exercise the wider matrix.
static int
test_linear_forward( void )
{
        const int rows = HERMES_NN_OUT, cols = HERMES_NN_HID;
        double    W[HERMES_NN_OUT * HERMES_NN_HID], b[HERMES_NN_OUT];
        double    x[HERMES_NN_HID];
        randomize( W, rows * cols, -0.5, 0.5 );
        randomize( b, rows, -0.5, 0.5 );
        randomize( x, cols, -1.0, 1.0 );

        struct hermes_nn nn;
        memset( &nn, 0, sizeof( nn ) );
        for ( int k = 0; k < rows * cols; k++ ) nn.w2[k] = (float)W[k];
        for ( int o = 0; o < rows; o++ ) nn.b2[o] = (float)b[o];

        float xf[HERMES_NN_HID], out[HERMES_NN_OUT];
        for ( int i = 0; i < cols; i++ ) xf[i] = (float)x[i];
        drive_linear( &nn, NN_PARAM_W2, NN_PARAM_B2, rows, cols, xf, out );

        double ref[HERMES_NN_OUT];
        ref_linear( W, b, x, rows, cols, ref );

        double maxerr = 0.0;
        for ( int o = 0; o < rows; o++ ) UPD( maxerr, out[o], ref[o] );
        return report( "linear-fwd", maxerr, FWD_TOL );
}

// Forward: SOFTMAX output matches the reference and sums to 1.
static int
test_softmax_forward( void )
{
        const int n = HERMES_NN_OUT;
        double    z[HERMES_NN_OUT];
        randomize( z, n, -2.0, 2.0 );

        struct hermes_nn nn;
        memset( &nn, 0, sizeof( nn ) );

        float zf[HERMES_NN_OUT], p[HERMES_NN_OUT];
        for ( int i = 0; i < n; i++ ) zf[i] = (float)z[i];
        drive_softmax( &nn, zf, n, p );

        double ref[HERMES_NN_OUT];
        ref_softmax( z, n, ref );

        double maxerr = 0.0, sum = 0.0;
        for ( int i = 0; i < n; i++ ) {
                UPD( maxerr, p[i], ref[i] );
                sum += (double)p[i];
        }
        UPD( maxerr, sum, 1.0 );  // probabilities must normalize
        return report( "softmax-fwd", maxerr, FWD_TOL );
}

// Backward: DLINEAR gradients (g_W, g_b, g_x) match central finite differences
// of L = <g, W x + b>.
static int
test_linear_backward( void )
{
        const int rows = HERMES_NN_OUT, cols = HERMES_NN_HID;
        double    W[HERMES_NN_OUT * HERMES_NN_HID], b[HERMES_NN_OUT];
        double    x[HERMES_NN_HID], g[HERMES_NN_OUT];
        randomize( W, rows * cols, -0.5, 0.5 );
        randomize( b, rows, -0.5, 0.5 );
        randomize( x, cols, -1.0, 1.0 );
        randomize( g, rows, -1.0, 1.0 );

        struct hermes_nn nn;
        memset( &nn, 0, sizeof( nn ) );  // also zeroes g_w2 / g_b2 accumulators
        for ( int k = 0; k < rows * cols; k++ ) nn.w2[k] = (float)W[k];
        for ( int o = 0; o < rows; o++ ) nn.b2[o] = (float)b[o];

        float xf[HERMES_NN_HID], gf[HERMES_NN_OUT], gx[HERMES_NN_HID];
        for ( int i = 0; i < cols; i++ ) xf[i] = (float)x[i];
        for ( int o = 0; o < rows; o++ ) gf[o] = (float)g[o];
        drive_dlinear( &nn, NN_PARAM_W2, NN_PARAM_B2, rows, cols, xf, gf, gx );

        double maxerr = 0.0;

        // dL/dW[o,i] vs g_w2.
        for ( int o = 0; o < rows; o++ ) {
                for ( int i = 0; i < cols; i++ ) {
                        double save = W[o * cols + i];
                        W[o * cols + i] = save + FD_EPS;
                        double Lp = loss_linear( W, b, x, g, rows, cols );
                        W[o * cols + i] = save - FD_EPS;
                        double Lm = loss_linear( W, b, x, g, rows, cols );
                        W[o * cols + i] = save;
                        double num = ( Lp - Lm ) / ( 2.0 * FD_EPS );
                        UPD( maxerr, nn.g_w2[o * cols + i], num );
                }
        }
        // dL/db[o] vs g_b2.
        for ( int o = 0; o < rows; o++ ) {
                double save = b[o];
                b[o]        = save + FD_EPS;
                double Lp   = loss_linear( W, b, x, g, rows, cols );
                b[o]        = save - FD_EPS;
                double Lm   = loss_linear( W, b, x, g, rows, cols );
                b[o]        = save;
                double num  = ( Lp - Lm ) / ( 2.0 * FD_EPS );
                UPD( maxerr, nn.g_b2[o], num );
        }
        // dL/dx[i] vs the pushed input gradient gx.
        for ( int i = 0; i < cols; i++ ) {
                double save = x[i];
                x[i]        = save + FD_EPS;
                double Lp   = loss_linear( W, b, x, g, rows, cols );
                x[i]        = save - FD_EPS;
                double Lm   = loss_linear( W, b, x, g, rows, cols );
                x[i]        = save;
                double num  = ( Lp - Lm ) / ( 2.0 * FD_EPS );
                UPD( maxerr, gx[i], num );
        }
        return report( "linear-bwd", maxerr, BWD_TOL );
}

// Backward: DSOFTMAX seed dL/dz matches central finite differences of
// L = -log(softmax(z)[action]) * reward, and logp = log probs[action].
static int
test_softmax_backward( void )
{
        const int    n      = HERMES_NN_OUT;
        const int    action = 1;
        const double reward = 1.5;
        double       z[HERMES_NN_OUT];
        randomize( z, n, -2.0, 2.0 );

        double p[HERMES_NN_OUT];
        ref_softmax( z, n, p );

        struct hermes_nn nn;
        memset( &nn, 0, sizeof( nn ) );

        float pf[HERMES_NN_OUT], dz[HERMES_NN_OUT], logp;
        for ( int i = 0; i < n; i++ ) pf[i] = (float)p[i];
        drive_dsoftmax( &nn, pf, n, action, (float)reward, dz, &logp );

        double maxerr = 0.0;
        UPD( maxerr, logp, log( p[action] ) );  // recorded log-prob
        for ( int i = 0; i < n; i++ ) {
                double save = z[i];
                z[i]        = save + FD_EPS;
                double Lp   = loss_softmax( z, n, action, reward );
                z[i]        = save - FD_EPS;
                double Lm   = loss_softmax( z, n, action, reward );
                z[i]        = save;
                double num  = ( Lp - Lm ) / ( 2.0 * FD_EPS );
                UPD( maxerr, dz[i], num );
        }
        return report( "softmax-bwd", maxerr, BWD_TOL );
}

int
main( void )
{
        srand( 1234 );

        int rc = 0;
        rc |= test_linear_forward( );
        rc |= test_softmax_forward( );
        rc |= test_linear_backward( );
        rc |= test_softmax_backward( );

        if ( rc ) {
                fprintf( stderr, "FAIL\n" );
                return 1;
        }
        printf( "OK\n" );
        return 0;
}
