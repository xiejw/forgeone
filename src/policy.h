#ifndef HERMES_POLICY_H
#define HERMES_POLICY_H

enum hermes_action {
        HERMES_ACTION_NONE  = 0,
        HERMES_ACTION_LEFT  = 1,
        HERMES_ACTION_RIGHT = 2,
};

const char *hermes_action_name( enum hermes_action act );

enum hermes_action hermes_random_policy( double pos, double speed );
enum hermes_action hermes_rev_policy( double pos, double speed );

#endif
