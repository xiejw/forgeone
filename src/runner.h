#ifndef HERMES_RUNNER_H
#define HERMES_RUNNER_H

#include "policy.h"

#define HERMES_MAX_STEPS  1000
#define HERMES_TRACK_HALF 30

typedef enum hermes_action (*hermes_policy_fn)(double pos, double speed);

// Runs one episode. Returns total reward. If verbose != 0, prints an ASCII
// track frame per step plus the final total, matching py/runner.py.
double hermes_run_episode(hermes_policy_fn policy, int verbose);

#endif
