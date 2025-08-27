#!/bin/bash

# =============================================================================
# Integrated Benchmark Script for vLLM Simple Serving
# =============================================================================

set -e  # Exit on any error

# Configuration - can be overridden via environment variables
MODEL=${MODEL:-/workspace/models/Llama-3.1-8B-Instruct}
GPU_ID=${GPU_ID:-0}
SERVER_PORT=${SERVER_PORT:-8027}

# Benchmark configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="results"
REQUEST_RATES=(1 2 3 4 5 6 7 8 9 10 11)

# Global variables
SERVER_PID=""

# Cleanup function
cleanup() {
    echo "Stopping everything…"
    trap - INT TERM        # prevent re-entrancy
    # Kill the server process
    if [[ -n $SERVER_PID ]]; then
        kill -9 $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
    fi
    exit 0
}

# Trap signals to ensure cleanup
trap cleanup INT TERM EXIT

# Function to wait for server to be ready
wait_for_server() {
    local port=$1
    local timeout_seconds=300
    local start_time=$(date +%s)

    echo "Waiting for server on port $port..."
    
    while true; do
        if curl -s "localhost:${port}/v1/completions" > /dev/null; then
            echo "Server on port $port is ready."
            return 0
        fi
        
        local now=$(date +%s)
        if (( now - start_time >= timeout_seconds )); then
            echo "Timeout waiting for server on port $port"
            return 1
        fi
        
        sleep 1
    done
}

# Function to start serving
start_serving() {
    echo "Starting vLLM server..."
    echo "Configuration:"
    echo "  Model: $MODEL"
    echo "  GPU ID: $GPU_ID"
    echo "  Port: $SERVER_PORT"
    echo ""
    
    # Start vLLM service
    CUDA_VISIBLE_DEVICES=$GPU_ID vllm serve \
        --model "$MODEL" \
        --tensor-parallel-size 1 \
        --seed 1024 \
        --dtype float16 \
        --max-model-len 10000 \
        --max-num-batched-tokens 10000 \
        --max-num-seqs 256 \
        --trust-remote-code \
        --gpu-memory-utilization 0.9 \
        --port $SERVER_PORT &
    SERVER_PID=$!
    
    # Wait for service to start
    sleep 10
    
    # Wait for server to be ready
    if ! wait_for_server $SERVER_PORT; then
        echo "Failed to start server on port $SERVER_PORT"
        cleanup
        exit 1
    fi
    
    echo "Server started successfully."
}

# Main execution
main() {
    echo "Starting integrated benchmark script for simple mode..."
    
    # Create results directory
    mkdir -p "$RESULTS_DIR"
    
    # Start the serving
    echo "Starting serving..."
    start_serving
    
    # Run benchmarks
    for REQUEST_RATE in "${REQUEST_RATES[@]}"; do
        NUM_PROMPTS=$((REQUEST_RATE * 300))
        echo "Running benchmark with request rate: $REQUEST_RATE, total prompts: $NUM_PROMPTS"
        
        # Run the benchmark
        vllm bench serve \
            --backend vllm \
            --model "$MODEL" \
            --endpoint /v1/completions \
            --dataset-name random \
            --random-input-len 512 \
            --random-output-len 64 \
            --ignore-eos \
            --metric-percentiles "90,95,99" \
            --seed 1024 \
            --trust-remote-code \
            --request-rate $REQUEST_RATE \
            --num_prompt $NUM_PROMPTS \
            --save-result \
            --save-detailed \
            --result-dir ./$RESULTS_DIR \
            --port $SERVER_PORT
            
        echo "Benchmark with request rate $REQUEST_RATE and $NUM_PROMPTS prompts completed."
        sleep 10
    done
    
    echo "All benchmarks completed."
    
    # Cleanup
    cleanup
}

# Run main function
main "$@"