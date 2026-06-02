# CLAUDE.md — hermes (C)

Minimal C port of the `py/` cart RL demo. Mirrors `env`, `RandomPolicy`,
`RevPolicy`, and the one-off episode runner. `NNPolicy` and REINFORCE training
are deferred until a torch-like engine lands.

## Coding Style
Read `../doc/lang.md`.

## Quick Start

```bash
make                # build .build/cart
make test           # build and run unit tests
make run            # run demo with random policy (default)
make run_rev        # run demo with rev policy
make RELEASE=1 all  # optimized build
make clean
```

## Coding Conventions

- All `#define` constants must have a comment explaining their purpose and units.
- Public constants go in `.h`; file-private constants go in `.c`.
- Ambiguous arguments (booleans, magic integers) at call sites must carry an
  inline name comment: `/*param_name=*/value`.

## Layout

- `env.h` / `env.c` — `struct hermes_env`, physics constants, per-tick step.
- `policy.h` / `policy.c` — `enum hermes_action`, `hermes_random_policy`,
  `hermes_rev_policy`.
- `runner.h` / `runner.c` — `hermes_run_episode` with ASCII track render.
- `main.c` — CLI: `./cart [random|rev]`.
- `test.c` — links the lib files (not `main.c`) and asserts reward bounds.

## Followups

- `hermes_nn` engine: tensors, linear, tanh, categorical, Adam, autograd.
- Port `NNPolicy` + REINFORCE trainer + eval harness on top of the engine.
- Wire `nn` and `nn_par` into the CLI.
