#!/bin/bash
#
# Expect the $1 is the build directory.
#
set -ex

if [ -z "$1" ]; then
    echo "Error: No argument provided."
    exit 1
fi

pushd $1

if [ -d "llama.cpp" ]; then
    echo "Error: Folder already exists." >&2
    popd
    exit 1
fi

git clone --depth 1 https://github.com/ggml-org/llama.cpp.git

cmake llama.cpp -B ./build   \
  -DBUILD_SHARED_LIBS=OFF    \
  -DGGML_CUDA=OFF            \
  -DLLAMA_BUILD_BORINGSSL=ON

cmake --build ./build --config Release -j --clean-first --target llama-cli llama-mtmd-cli llama-server llama-gguf-split

popd

# Test the server:
# ```
# REQUEST='{"model":"default","messages":[{"role":"user","content":"Hello!"}]}'
#
# echo "==> Request" && echo "${REQUEST}" | jq -C .
#
# echo "==> Response" && curl -s http://127.0.0.1:8080/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -H "Authorization: Bearer sk-no-key-required" \
#   -d "${REQUEST}" \
#   | jq -C .
# ```
