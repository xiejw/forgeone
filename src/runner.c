#include "runner.h"

#include <math.h>
#include <stdio.h>

#include "env.h"

#define HERMES_TRACK_CELLS (2 * HERMES_TRACK_HALF + 1)

#define CYAN  "\033[36m"
#define RESET "\033[0m"

// === --- Rendering ---------------------------------------------------- ===

static const char *wind_symbol(int wind_dir)
{
        if (wind_dir > 0) return "wind-->";
        if (wind_dir < 0) return "<--wind";
        return "  ...  ";
}

static void render(double pos, char *out)
{
        double scale = (double)HERMES_TRACK_HALF / HERMES_POSITION_LIMIT;
        long   idx   = (long)lround(pos * scale) + HERMES_TRACK_HALF;
        if (idx < 0)                       idx = 0;
        if (idx > 2 * HERMES_TRACK_HALF)   idx = 2 * HERMES_TRACK_HALF;

        out[0] = '[';
        for (int i = 0; i < HERMES_TRACK_CELLS; i++) out[1 + i] = '-';
        out[1 + HERMES_TRACK_HALF] = '|';
        out[1 + idx]               = '#';
        out[1 + HERMES_TRACK_CELLS] = ']';
        out[2 + HERMES_TRACK_CELLS] = '\0';
}

static void print_frame(int step, double pos, double speed,
                        enum hermes_action act, int wind_dir)
{
        char track[3 + HERMES_TRACK_CELLS];
        render(pos, track);
        printf("step %3d %s pos=%+6.1f speed=%+5.1f act=%-5s "
               CYAN "%s" RESET "\n",
               step, track, pos, speed, hermes_action_name(act),
               wind_symbol(wind_dir));
}

// === --- Episode loop ------------------------------------------------- ===

double hermes_run_episode(hermes_policy_fn policy, int verbose)
{
        struct hermes_env env;
        hermes_env_init(&env);

        double total_reward = 0.0;
        for (int step = 0; step < HERMES_MAX_STEPS; step++) {
                double pos, speed;
                hermes_env_obs(&env, &pos, &speed);
                enum hermes_action act = policy(pos, speed);

                double reward;
                int    game_over = hermes_env_action(&env, act, &reward);
                total_reward += reward;

                if (verbose) {
                        double next_pos, next_speed;
                        hermes_env_obs(&env, &next_pos, &next_speed);
                        print_frame(step + 1, next_pos, next_speed, act,
                                    env.wind_dir);
                }
                if (game_over) {
                        if (verbose) printf("game over\n");
                        break;
                }
        }
        if (verbose) printf("total reward: %.1f\n", total_reward);
        return total_reward;
}
