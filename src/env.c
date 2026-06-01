#include "env.h"

#include <assert.h>
#include <stdlib.h>

// === --- Random helpers ----------------------------------------------- ===

static double rand_unit(void)
{
        return (double)rand() / (double)RAND_MAX;
}

static double rand_uniform(double lo, double hi)
{
        return lo + rand_unit() * (hi - lo);
}

static int rand_dir(void)
{
        return (rand() & 1) ? 1 : -1;
}

// === --- Per-tick physics --------------------------------------------- ===

static void apply_wind(struct hermes_env *env)
{
        if (env->wind_dir == 0) return;
        env->speed += (double)env->wind_dir * env->wind_strength * HERMES_DT;
        env->wind_strength -= HERMES_WIND_DECAY * HERMES_DT;
        if (env->wind_strength < 0.0) env->wind_strength = 0.0;
}

static void apply_action(struct hermes_env *env, enum hermes_action act)
{
        if      (act == HERMES_ACTION_RIGHT) env->speed += HERMES_ACTION_ACCELERATION * HERMES_DT;
        else if (act == HERMES_ACTION_LEFT)  env->speed -= HERMES_ACTION_ACCELERATION * HERMES_DT;
}

static void apply_friction(struct hermes_env *env)
{
        if (env->speed > 0.0) {
                env->speed -= HERMES_SPEED_DECAY * HERMES_DT;
                if (env->speed < 0.0) env->speed = 0.0;
        } else if (env->speed < 0.0) {
                env->speed += HERMES_SPEED_DECAY * HERMES_DT;
                if (env->speed > 0.0) env->speed = 0.0;
        }
}

static void advance_position(struct hermes_env *env)
{
        env->position += env->speed * HERMES_DT;
}

static void tick_wind_timer(struct hermes_env *env)
{
        if (env->wind_dir == 0) return;
        env->wind_remaining -= HERMES_DT;
        if (env->wind_remaining <= 0.0) {
                env->wind_dir = 0;
                env->wind_remaining = 0.0;
        }
}

static void maybe_restart_wind(struct hermes_env *env)
{
        if (env->wind_dir != 0) return;
        if (rand_unit() >= HERMES_WIND_RESTART_CHANCE) return;
        env->wind_dir       = rand_dir();
        env->wind_strength  = HERMES_WIND_ACCELERATION;
        env->wind_remaining = rand_uniform(HERMES_WIND_MIN_DURATION,
                                           HERMES_WIND_MAX_DURATION);
}

// === --- Public API --------------------------------------------------- ===

void hermes_env_init(struct hermes_env *env)
{
        env->position       = 0.0;
        env->speed          = 0.0;
        env->wind_dir       = 0;
        env->wind_remaining = 0.0;
        env->wind_strength  = 0.0;
        env->game_over      = 0;
}

void hermes_env_obs(const struct hermes_env *env,
                    double *pos_out, double *speed_out)
{
        *pos_out   = env->position;
        *speed_out = env->speed;
}

int hermes_env_action(struct hermes_env *env,
                      enum hermes_action act,
                      double *reward_out)
{
        assert(!env->game_over);

        apply_wind(env);
        apply_action(env, act);
        apply_friction(env);
        advance_position(env);
        tick_wind_timer(env);
        maybe_restart_wind(env);

        double abs_pos = env->position < 0.0 ? -env->position : env->position;
        if (abs_pos >= HERMES_POSITION_LIMIT) {
                env->game_over = 1;
                *reward_out    = 0.0;
                return 1;
        }
        *reward_out = HERMES_REWARD_PER_STEP;
        return 0;
}
