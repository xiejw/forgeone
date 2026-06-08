BUILD  = .build
SHELL := /bin/bash

LLAMA_SERVER_BUILD_DIR = ${BUILD}/server
LLAMA_SERVER           = ${LLAMA_SERVER_BUILD_DIR}/server/build/llama_server

.PHONY: llama_server_install llama_server_run

compile:
	cd src ; cargo check

run:
	cd src ; cargo run

run_rev:
	cd src ; cargo run -- rev

fmt:
	cd src ; cargo fmt

test:
	cd src ; cargo test --lib --bins

clean_rs:
	cd src ; cargo clean

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

