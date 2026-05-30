HF_MODEL=unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_XL
./build/bin/llama-server     -hf ${HF_MODEL}    --host 127.0.0.1 --port 8080 --api-key sk-no-key-required
