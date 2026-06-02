#include <stdio.h>
#include <stdlib.h>

#include "env.h"
#include "policy.h"
#include "runner.h"

// Mirrors py/test_policy.py: reward must land in [1, HERMES_MAX_STEPS].
// Each surviving step earns 1.0; the final game-over step earns 0.0. So if
// the cap is hit, step_count == reward; otherwise step_count == reward + 1.

static int step_count;

static enum hermes_action
counting_random( double pos, double speed )
{
        step_count++;
        return hermes_random_policy( pos, speed );
}

static enum hermes_action
counting_rev( double pos, double speed )
{
        step_count++;
        return hermes_rev_policy( pos, speed );
}

static int
check_policy( const char *name, hermes_policy_fn fn )
{
        struct hermes_env env;
        hermes_env_init( &env );
        defer { hermes_env_deinit( &env ); }

        step_count    = 0;
        double reward = hermes_run_episode( &env, fn, 0 );

        if ( !( reward >= 1.0 && reward <= (double)HERMES_MAX_STEPS ) ) {
                fprintf( stderr, "[%s] reward %.1f out of [1, %d]\n", name,
                         reward, HERMES_MAX_STEPS );
                return 1;
        }
        if ( step_count < 1 || step_count > HERMES_MAX_STEPS ) {
                fprintf( stderr, "[%s] step_count %d out of [1, %d]\n", name,
                         step_count, HERMES_MAX_STEPS );
                return 1;
        }
        int reward_int = (int)reward;
        if ( step_count != reward_int && step_count != reward_int + 1 ) {
                fprintf( stderr,
                         "[%s] step_count %d not in {reward, reward+1} (%d)\n",
                         name, step_count, reward_int );
                return 1;
        }
        printf( "[%s] reward=%.1f steps=%d OK\n", name, reward, step_count );
        return 0;
}

int
main( void )
{
        srand( 0 );

        int rc = 0;
        rc |= check_policy( "random", counting_random );
        rc |= check_policy( "rev", counting_rev );
        if ( rc ) {
                fprintf( stderr, "FAIL\n" );
                return 1;
        }
        printf( "OK\n" );
        return 0;
}
