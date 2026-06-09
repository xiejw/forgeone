# Rust — Coding Style & Conventions

Applies to the Rust crate at the repo root (sources under `src/`).

---

## Toolchain & dependencies

- **Edition 2024.** Build with stable `cargo`.
- **std-only — no third-party crates.** Anything that would normally be a
  dependency is hand-rolled (e.g. the PRNG in `rng.rs`), keeping the crate
  self-contained. Add a dependency only if the project explicitly calls for it.
- Code must build clean under `cargo build`, `cargo test`, and
  `cargo clippy --all-targets` (zero warnings), and be `cargo fmt`-clean.
- Build artifacts go to `.build/` (configured in `.cargo/config.toml`), not the
  default `target/`.

## Module layout

- **One module per concern, one file per module.** Each C-style "translation
  unit" maps to a file: `env.rs`, `policy.rs`, `runner.rs`, `nn.rs`, `rng.rs`.
  `lib.rs` declares the modules; `main.rs` is the CLI entry point only.
- Keep foundational/shared infrastructure in its own module (e.g. `rng.rs`),
  used by the domain modules rather than duplicated.
- Put the binary's logic in the library; `main.rs` just wires arguments to it.

## Naming

- Standard Rust casing: `UpperCamelCase` types/traits/enum variants,
  `snake_case` functions/methods/locals/modules, `SCREAMING_SNAKE_CASE`
  constants.
- Methods whose first argument is the owning type take `self`/`&self`/`&mut self`
  (no manual "self" parameter).
- Return computed values; do **not** use C-style out-parameters. For multiple
  results use a tuple or a small named struct (e.g. `env::Step { reward, done }`).

## Code style

- Each logical concern gets its own function/method. No monolithic functions —
  e.g. `Env::step` delegates to `apply_wind`, `apply_friction`, etc.
- Doc comments: `//!` for module-level docs (top of file), `///` for public
  items. Explain intent and units, not the obvious.
- Section banners group related items within a file:
  ```rust
  // === --- Section name ------------------------------------------------- ===
  ```
- One blank line between item definitions.
- Prefer borrows over copies: hold a borrow only as long as needed, and prefer
  field-disjoint borrows (e.g. `&self.forward` + `&mut self.net`) over cloning to
  satisfy the borrow checker. Avoid gratuitous `.clone()` / `.to_vec()`.

## Constants & magic numbers

- No magic numbers: define named constants at the top of the module, each with a
  comment giving its purpose and units (e.g. the physics constants in `env.rs`).
- Public constants are `pub const`; module-private ones are plain `const`.

## Error handling & invariants

- Shapes and topology are statically known, so internal invariants are checked
  with `assert!` / `debug_assert!` (e.g. `assert!(!self.game_over)` in
  `Env::step`, length checks in the VM ops). There is no error-stack machinery —
  this is a small, fully-static demo.
- Use `expect("…")` with a message that states the invariant when unwrapping
  something that "cannot fail" by construction (e.g. stack/tape pops).
- Reserve `panic!` for genuinely unreachable / programmer-error cases.

## Tests

- Unit tests live in a `#[cfg(test)] mod tests` block at the bottom of the
  module they cover (`use super::*;`).
- Test by behavior and bounds, not by matching incidental details (e.g. the
  random policy is checked for rough uniformity, not an exact RNG sequence; the
  network's gradients are checked against finite differences).
