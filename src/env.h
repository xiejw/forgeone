#ifndef HERMES_ENV_H
#define HERMES_ENV_H

#include "policy.h"

// === --- Physics constants -------------------------------------------- ===
//
// Mirror py/env.py so behavior matches the reference implementation.

#define HERMES_POSITION_LIMIT       40.0
#define HERMES_SPEED_DECAY           1.0
#define HERMES_WIND_ACCELERATION     3.0
#define HERMES_WIND_DECAY            0.5
#define HERMES_ACTION_ACCELERATION   2.0
#define HERMES_WIND_RESTART_CHANCE   0.5
#define HERMES_WIND_MIN_DURATION     1.0
#define HERMES_WIND_MAX_DURATION     5.0
#define HERMES_DT                    1.0
#define HERMES_REWARD_PER_STEP       1.0

// === --- Env type and API --------------------------------------------- ===

struct hermes_env {
        double position;
        double speed;
        int    wind_dir;        // -1, 0, +1
        double wind_remaining;
        double wind_strength;
        int    game_over;
};

void hermes_env_init(struct hermes_env *env);

void hermes_env_obs(const struct hermes_env *env,
                    double *pos_out, double *speed_out);

// Advances the simulation by one tick. Writes the step reward into
// *reward_out. Returns 1 if the env just transitioned to game_over, else 0.
// Calling on an already-over env is a programmer error.
int hermes_env_action(struct hermes_env *env,
                      enum hermes_action act,
                      double *reward_out);

#endif
