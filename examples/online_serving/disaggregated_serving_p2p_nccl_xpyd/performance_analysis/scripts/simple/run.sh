#!/bin/bash

# =============================================================================
# Main Control Script for vLLM Simple Serving
# =============================================================================

set -xe  # Exit on any error

# Configuration - can be overridden via environment variables
MODEL=${MODEL:-/workspace/models/Llama-3.1-8B-Instruct}
GPU_ID=${GPU_ID:-0}
SERVER_PORT=${SERVER_PORT:-8027}

# Benchmark configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUEST_RATES=${REQUEST_RATES:-"0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5"}

# List of benchmark scripts to run
BENCH_SCRIPTS=(
    "random-512-64.sh"
    "sharegpt.sh"
    "gsm8k.sh"
    "human_eval.sh"
)

# Global variables
SERVE_PID=""
BENCH_PID=""

# Cleanup function
cleanup() {
    echo "Stopping everything…"
    trap - INT TERM        # prevent re-entrancy
    
    # Kill the serve process
    if [[ -n $SERVE_PID ]]; then
        echo "Killing serve process $SERVE_PID..."
        if kill -0 "$SERVE_PID" 2>/dev/null; then
            kill -TERM "$SERVE_PID" 2>/dev/null || true
            
            # Wait for process to terminate gracefully
            echo "Waiting for serve process to terminate..."
            timeout 10 wait "$SERVE_PID" 2>/dev/null || true
            
            # Force kill if still running
            if kill -0 "$SERVE_PID" 2>/dev/null; then
                echo "Force killing serve process $SERVE_PID..."
                kill -KILL "$SERVE_PID" 2>/dev/null || true
            fi
        fi
    fi
    
    # Kill the bench process
    if [[ -n $BENCH_PID ]]; then
        echo "Killing bench process $BENCH_PID..."
        if kill -0 "$BENCH_PID" 2>/dev/null; then
            kill -TERM "$BENCH_PID" 2>/dev/null || true
            
            # Wait for process to terminate gracefully
            echo "Waiting for bench process to terminate..."
            timeout 10 wait "$BENCH_PID" 2>/dev/null || true
            
            # Force kill if still running
            if kill -0 "$BENCH_PID" 2>/dev/null; then
                echo "Force killing bench process $BENCH_PID..."
                kill -KILL "$BENCH_PID" 2>/dev/null || true
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
    echo "Starting serving components..."
    
    # Export environment variables for serve script
    export MODEL
    export GPU_ID
    export SERVER_PORT
    
    # Start serve script in background
    "$SCRIPT_DIR/serve.sh" &
    SERVE_PID=$!
    
    # Wait for server to be ready
    if ! wait_for_server $SERVER_PORT; then
        echo "Failed to start server on port $SERVER_PORT"
        cleanup
        exit 1
    fi
    
    echo "Serving components started with PID $SERVE_PID."
}

# Function to run benchmarks
run_benchmarks() {
    echo "Running benchmarks..."
    
    # Export environment variables for bench script
    export REQUEST_RATES
    export SERVER_PORT
    export MODEL_PATH="$MODEL"
    
    # Start bench script
    "$SCRIPT_DIR/$BENCH_SCRIPT"
    
    echo "Benchmarks completed."
}

# Main execution
main() {
    echo "Starting main control script for simple mode..."
    
    # Create results directory
    mkdir -p "$SCRIPT_DIR/results"
    
    # Start serving
    start_serving
    
    # Wait a bit more for all services to be ready
    sleep 10
    
    # Run benchmarks with different datasets
    for BENCH_SCRIPT in "${BENCH_SCRIPTS[@]}"; do
        echo "Running benchmarks with script: $BENCH_SCRIPT"
        run_benchmarks
    done
    
    # Cleanup
    cleanup
}

# Run main function
main "$@"