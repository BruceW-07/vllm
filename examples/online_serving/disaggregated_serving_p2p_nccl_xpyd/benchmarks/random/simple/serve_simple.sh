#!/bin/bash

# Define cleanup function to clean up resources when the script exits
cleanup() {
    echo "Stopping vLLM server..."
    # Stop the entire process group to ensure all child processes are terminated
    trap - INT TERM        # prevent re-entrancy
    kill -- -$$            # negative PID == "this whole process-group"
    wait                   # reap children so we don't leave zombies
    echo "vLLM server stopped."
    exit 0
}

# Set up signal traps to ensure cleanup function is called on script exit
trap cleanup INT TERM EXIT

# Start vLLM service
CUDA_VISIBLE_DEVICES=0 vllm serve \
    --model "/workspace/models/Llama-3.1-8B-Instruct" \
    --tensor-parallel-size 1 \
    --seed 1024 \
    --dtype float16 \
    --max-model-len 10000 \
    --max-num-batched-tokens 10000 \
    --max-num-seqs 256 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --port 8027 &

# Wait for service to start
sleep 10

# Keep the script running
while true; do
    sleep 60
done