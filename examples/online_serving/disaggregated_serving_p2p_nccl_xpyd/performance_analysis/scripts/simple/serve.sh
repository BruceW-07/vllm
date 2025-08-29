#!/bin/bash

# =============================================================================
# Serve Script for vLLM Simple Serving
# =============================================================================

set -xe  # Exit on any error

# Benchmark configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Global variables
SERVER_PID=""

# Cleanup function
cleanup() {
    echo "Stopping everything…"
    trap - INT TERM        # prevent re-entrancy
    
    # Kill the server process
    if [[ -n $SERVER_PID ]]; then
        echo "Killing server process $SERVER_PID..."
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            kill -TERM "$SERVER_PID" 2>/dev/null || true
            
            # Wait for process to terminate gracefully
            echo "Waiting for server process to terminate..."
            timeout 10 wait "$SERVER_PID" 2>/dev/null || true
            
            # Force kill if still running
            if kill -0 "$SERVER_PID" 2>/dev/null; then
                echo "Force killing server process $SERVER_PID..."
                kill -KILL "$SERVER_PID" 2>/dev/null || true
            fi
        fi
    fi
    
    # Additional cleanup for any remaining vllm processes
    echo "Cleaning up any remaining vllm processes..."
    pkill -f "vllm serve" 2>/dev/null || true
    
    # Wait a bit for processes to fully terminate
    sleep 2
    
    echo "Cleanup completed."
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
    CUDA_VISIBLE_DEVICES=$GPU_ID vllm serve "$MODEL" \
        --no-enable-prefix-caching \
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
    echo "Starting serve script for simple mode..."
    
    # Start the serving
    echo "Starting serving..."
    start_serving
    
    # Keep the script running
    echo "Serve component is running. Press Ctrl+C to stop."
    wait
}

# Run main function
main "$@"