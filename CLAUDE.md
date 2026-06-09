## Project

A tiny cart RL demo. The crate is the repo root — a standard Cargo project:
manifest at `Cargo.toml`, sources under `src/` (`env`, `policy`,
`runner`, `nn`, `rng`, and the `cart` binary in `main.rs`).

Build/run/test from the repo root via the `Makefile` (`make compile`, `make run`,
`make run_rev`, `make test`, `make fmt`).

## Coding Style

Rust coding style and conventions: read `./doc/lang.md`.

## Design

Project design notes (the `nn` stack-VM policy): read `./doc/DESIGN.md`.
