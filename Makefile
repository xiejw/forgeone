BUILD  = .build
SHELL := /bin/bash

LLAMA_SERVER_BUILD_DIR = ${BUILD}/server
LLAMA_SERVER           = ${LLAMA_SERVER_BUILD_DIR}/server/build/llama_server

.PHONY: llama_server_install llama_server_run

compile:
	cd rs ; cargo check

run:
	cd rs ; cargo run

test:
	cd rs ; cargo test --lib --bins

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

