BUILD  = .build
SHELL := /bin/bash

LLAMA_SERVER_BUILD_DIR = ${BUILD}/server
LLAMA_SERVER           = ${LLAMA_SERVER_BUILD_DIR}/server/build/llama_server

.PHONY: llama_server_install llama_server_run

compile:
	cargo check

run:
	cargo run

run_rev:
	cargo run -- rev

run_nn:
	cargo run -- nn

fmt:
	cargo fmt

test:
	cargo test --lib

clean_rs:
	cargo clean

clean: clean_rs

llama_server_install: ${LLAMA_SERVER}


# === --- House Keeping --- ===

${BUILD}:
	mkdir -p $@

clean:
	rm -rf ${BUILD}

# === --- Llama Server --- ===

${LLAMA_SERVER_BUILD_DIR}:
	mkdir -p $@

${LLAMA_SERVER}: ${LLAMA_SERVER_BUILD_DIR}
	sh scripts/llama_server_install.sh ${LLAMA_SERVER_BUILD_DIR}

