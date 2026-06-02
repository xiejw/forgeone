#ifndef HERMES_RUNNER_H
#define HERMES_RUNNER_H

#include "env.h"
#include "policy.h"

// Maximum number of steps before the episode is force-terminated.
#define HERMES_MAX_STEPS 1000

typedef enum hermes_action ( *hermes_policy_fn )( double pos, double speed );

// Runs one episode on a caller-owned env (must be freshly hermes_env_init'd).
// Returns total reward. If verbose != 0, prints an ASCII track frame per step
// plus the final total, matching py/runner.py.
double hermes_run_episode( struct hermes_env *env, hermes_policy_fn policy,
                           int verbose );

#endif
