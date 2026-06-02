#ifndef HERMES_ENV_H
#define HERMES_ENV_H

#include "policy.h"

// === --- Physics Constants -------------------------------------------- ===
//

// Cart ends the game when |position| reaches this many units.
#define HERMES_POSITION_LIMIT 40.0
// Friction: speed moves toward zero by this much per second.
#define HERMES_SPEED_DECAY 1.0
// Wind's initial strength (units/s) when a wind episode starts.
#define HERMES_WIND_ACCELERATION 3.0
// Wind strength weakens this much per second, floored at 0.
#define HERMES_WIND_DECAY 0.5
// Left/right action adds this many units/s to speed.
#define HERMES_ACTION_ACCELERATION 2.0
// Probability that wind restarts after a wind episode ends.
#define HERMES_WIND_RESTART_CHANCE 0.5
// Shortest a wind episode can last (seconds).
#define HERMES_WIND_MIN_DURATION 1.0
// Longest a wind episode can last (seconds).
#define HERMES_WIND_MAX_DURATION 5.0
// Simulation time step (seconds per action call).
#define HERMES_DT 1.0
// Reward earned each surviving step.
#define HERMES_REWARD_PER_STEP 1.0

// === --- Env type and API --------------------------------------------- ===

struct hermes_env {
        double position;
        double speed;
        int    wind_dir;  // -1, 0, +1
        double wind_remaining;
        double wind_strength;
        int    game_over;
};

void hermes_env_init( struct hermes_env *env );
void hermes_env_deinit( struct hermes_env *env );

void hermes_env_obs( const struct hermes_env *env, double *pos_out,
                     double *speed_out );

// Advances the simulation by one tick. Writes the step reward into
// *reward_out. Returns 1 if the env just transitioned to game_over, else 0.
// Calling on an already-over env is a programmer error.
[[nodiscard]] int hermes_env_action( struct hermes_env *env,
                                     enum hermes_action act,
                                     double            *reward_out );

#endif
