# C Projects — Coding Style & Conventions

Applies to all C projects under this directory.

---

## Naming

- All **public** symbols (functions, types, macros): `forge_` prefix, `snake_case`.
  - No `typedef` for structs — declare as `struct snake_case_name`.
  - **Free functions** (no implicit "self"): `verb_noun` — e.g., `forge_convert_html`, `forge_parse_token`.
  - **Object-like functions** (first arg is the owning struct): `noun_verb` — e.g., `forge_err_init`, `forge_err_emit`.
- **Internal / static** symbols: `snake_case`, no prefix.
- Constants / macros: `UPPER_SNAKE_CASE`.

**hermes namespace** — ML data structures, algorithm, and functions
use the `hermes_` prefix (not `forge_`). Constants use the `HERMES_` prefix.
Infrastructure utilities (err_stack) keep the `forge_` prefix as usual.

- **forge_ namespace owns all foundational infrastructure** — error handling,
  parallel utilities, and similar shared data structures always use the `forge_`
  prefix, even when used by `hermes_` code.  The `hermes_` namespace is reserved
  for ML domain types and algorithms only.
- **Output parameters** (pointer args through which a function returns a computed
  value to the caller) must be named with the `_out` suffix — e.g., `size_t
  *n_out`, `uint32_t *val_out`, `void *dst_out`.  "In-out" self/object parameters
  are exempt.

## Code Style

- Standard: **C99** (`-std=c99`).
- Compiler flags: `-Wall -Werror -pedantic -Wextra -Wfatal-errors -Wconversion`.
- In Makefile, if `RELEASE` is defined, compiler flags should have
  `-O2 -DNDEBUG -march=native -flto -ffast-math` as well.
- Each logical concern gets its own function. No monolithic functions.
- No third-party libraries — stdlib only unless the project explicitly states otherwise.
- One blank line between top-level definitions. Section banners for logical groups:
  ```c
  // === --- Section name ----------------------------------------------- ===
  //
  ```

## Constants and Magic Numbers

All constants should be defined at top level to avoid magic numbers. And for
fixed size array, it is better to check the input length to report error.

## Error Handling

All fallible functions use this pattern:

```c
/* Return 0 on success, 1 on error. Errors are written into stk. */
int forge_foo(..., struct err_stack *stk);
```

### struct err_stack

Dynamic char buffer that accumulates error messages. Lives on the caller's stack.

```c
struct err_stack {
    char  *buf;
    size_t len;
    size_t cap;
};

void        forge_err_init(struct err_stack *stk);   /* zero-init */
void        forge_err_deinit(struct err_stack *stk); /* free heap buf */
void        forge_err_emit(struct err_stack *stk, const char *fmt, ...);
const char *forge_err_get(const struct err_stack *stk); /* NULL if empty */
```

### forge_err_emit — indicators and newlines

Each `forge_err_emit` call automatically appends `\n` and prepends a line indicator:

- **First call** (root cause): prefixed with `✗ `
- **Subsequent calls** (diagnostic notes added by callers): prefixed with `↳ `

The buffer therefore reads root-cause-first, with higher-level context below:

```
✗ fopen: no such file or directory
↳ failed to load template file "layout.html"
↳ render aborted
```

Implementation: checks `stk->len == 0` to choose the prefix. Uses `vsnprintf` into a doubling buffer (initial cap 256). If `realloc` fails, the message is silently truncated — acceptable for an error-reporting path.

Usage:

```c
int rc = 0;
struct err_stack stk = {0};
forge_err_init(&stk);

if (forge_foo(&stk)) {
    fprintf(stderr, "%s", forge_err_get(&stk));
    rc = 1;
    goto exit;
}

exit:
    forge_err_deinit(&stk);
    return rc;
```

## Resource Cleanup

Use `goto exit` to consolidate cleanup.  Every resource is freed in a single
`exit:` label at the end of the function.

```c
/* Pattern for int-returning functions with heap/file resources: */
int forge_foo(..., struct err_stack *stk)
{
    int   rc   = 0;
    FILE *f    = NULL;
    char *buf  = NULL;

    f = fopen(...);
    if (!f) { forge_err_emit(stk, "..."); rc = 1; goto exit; }

    buf = malloc(...);
    if (!buf) { forge_err_emit(stk, "..."); rc = 1; goto exit; }

    /* ... normal work ... */

exit:
    if (f) fclose(f);
    free(buf);        /* free(NULL) is safe */
    return rc;
}

/* Pattern for pointer-returning functions — use ok flag: */
static char *make_thing(...)
{
    int    ok  = 0;
    FILE  *f   = NULL;
    char  *buf = NULL;

    f = fopen(...); if (!f) { forge_err_emit(...); goto exit; }
    buf = malloc(...); if (!buf) { forge_err_emit(...); goto exit; }
    /* ... fill buf ... */
    ok = 1;

exit:
    if (f) fclose(f);
    if (!ok) { free(buf); buf = NULL; }
    return buf;
}
```


### Severity levels

| Situation | Action |
|---|---|
| Fatal (malloc fail, file open fail) | `forge_err_emit` + `return 1` immediately |
| Non-fatal parse/format error | `forge_err_emit` + continue; caller checks at the end |


## Build

Each project has a `Makefile` with at minimum:

```makefile
all:    # builds the main binary into .build/
test:   # compiles and runs test binary
clean:  # rm -rf .build
```

Test binaries link `md2html.c` (or equivalent lib file) directly — **not** `main.c`.
POSIX functions in tests require `-D_POSIX_C_SOURCE=200809L`.

## File Layout

```
project/
  foo.h       -- public types and API
  foo.c       -- implementation (static internals + public functions)
  main.c      -- CLI / entry point only
  test.c      -- test cases
  Makefile
  CLAUDE.md   -- project-specific notes
```
