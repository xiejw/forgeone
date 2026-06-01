#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "env.h"
#include "policy.h"
#include "runner.h"

// `defer { hermes_env_deinit( &env ); }` once clang ships the C defer TS
// (no `-fdefer-ts` flag in clang 21 yet). cleanup attribute matches the
// semantics: hermes_env_deinit(&env) runs when env leaves scope.
#define defer_cleanup( F ) __attribute__( ( cleanup( F ) ) )

int
main( int argc, char **argv )
{
        srand( (unsigned)time( NULL ) );

        const char      *name = ( argc > 1 ) ? argv[1] : "random";
        hermes_policy_fn fn   = NULL;
        if ( strcmp( name, "random" ) == 0 )
                fn = hermes_random_policy;
        else if ( strcmp( name, "rev" ) == 0 )
                fn = hermes_rev_policy;
        else {
                fprintf( stderr, "unknown policy: %s (expected random|rev)\n",
                         name );
                return 1;
        }

        defer_cleanup( hermes_env_deinit ) struct hermes_env env;
        hermes_env_init( &env );

        hermes_run_episode( &env, fn, 1 );
        return 0;
}
