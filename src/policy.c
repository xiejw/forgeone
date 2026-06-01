#include "policy.h"

#include <stdlib.h>

const char *
hermes_action_name( enum hermes_action act )
{
        switch ( act ) {
        case HERMES_ACTION_NONE:
                return "none";
        case HERMES_ACTION_LEFT:
                return "left";
        case HERMES_ACTION_RIGHT:
                return "right";
        }
        return "?";
}

enum hermes_action
hermes_random_policy( double pos, double speed )
{
        (void)pos;
        (void)speed;
        return ( enum hermes_action )( rand( ) % 3 );
}

enum hermes_action
hermes_rev_policy( double pos, double speed )
{
        (void)speed;
        if ( pos > 0.0 ) return HERMES_ACTION_LEFT;
        if ( pos < 0.0 ) return HERMES_ACTION_RIGHT;
        return HERMES_ACTION_NONE;
}
