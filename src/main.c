#include <stddefer.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "env.h"
#include "policy.h"
#include "runner.h"

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

        struct hermes_env env;
        hermes_env_init( &env );
        defer { hermes_env_deinit( &env ); }

        hermes_run_episode( &env, fn, /*verbose=*/1 );
        return 0;
}
