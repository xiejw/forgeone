#ifndef HERMES_POLICY_H
#define HERMES_POLICY_H

// Index ordering must match py/policy.py ACTIONS so a future NN head's
// output dim lines up cell-for-cell.
enum hermes_action {
        HERMES_ACTION_NONE  = 0,
        HERMES_ACTION_LEFT  = 1,
        HERMES_ACTION_RIGHT = 2,
};

const char *hermes_action_name( enum hermes_action act );

enum hermes_action hermes_random_policy( double pos, double speed );
enum hermes_action hermes_rev_policy( double pos, double speed );

#endif
