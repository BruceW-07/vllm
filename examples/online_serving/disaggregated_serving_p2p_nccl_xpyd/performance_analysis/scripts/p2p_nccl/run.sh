#!/bin/bash

# =============================================================================
# Main Control Script for vLLM Disaggregated Serving with P2P NCCL
# =============================================================================

set -xe  # Exit on any error

# Configuration - can be overridden via environment variables
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-1200}

# Default 1P3D configuration (1 Prefill + 3 Decode)
PREFILL_GPUS=${PREFILL_GPUS:-0}
DECODE_GPUS=${DECODE_GPUS:-1}
PREFILL_PORTS=${PREFILL_PORTS:-20003}
DECODE_PORTS=${DECODE_PORTS:-20005}

# Proxy configuration
PROXY_SERVICE_DISCOVERY_PORT=${PROXY_SERVICE_DISCOVERY_PORT:-30001}
PROXY_APP_PORT=${PROXY_APP_PORT:-10001}

# Benchmark configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROXY_SCRIPT="$SCRIPT_DIR/../../../disagg_proxy_p2p_nccl_xpyd.py"
REQUEST_RATES=${REQUEST_RATES:-"1 2 3 4 5 6 7 8 9 10 11"}

# List of benchmark scripts to run
BENCH_SCRIPTS=(
    "random-512-64.sh"
    "sharegpt.sh"
    "gsm8k.sh"
    "human_eval.sh"
)

# Default model path - update this to your actual model path
MODEL_PATH="/workspace/models/Llama-3.1-8B-Instruct"

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
    pkill -f "disagg_proxy_p2p_nccl_xpyd.py" 2>/dev/null || true
    
    # Wait a bit for processes to fully terminate
    sleep 2
    
    echo "Cleanup completed."
}

# Trap signals to ensure cleanup
trap cleanup INT TERM EXIT

# Function to check if Python library is installed
ensure_python_library_installed() {
    echo "Checking if $1 is installed..."
    if ! python3 -c "import $1" > /dev/null 2>&1; then
        echo "$1 is not installed. Please install it via pip install $1."
        exit 1
    else
        echo "$1 is installed."
    fi
}

# Function to check if command is available
ensure_command_available() {
    echo "Checking if $1 is available..."
    if ! command -v "$1" > /dev/null 2>&1; then
        echo "$1 is not installed. Please install it."
        exit 1
    else
        echo "$1 is available."
    fi
}

# Function to check dependencies
check_dependencies() {
    echo "Checking dependencies..."
    ensure_python_library_installed pandas
    ensure_python_library_installed datasets
    ensure_python_library_installed vllm
    ensure_python_library_installed quart
    ensure_command_available bc
    ensure_command_available curl
    echo "All dependencies are satisfied."
}

# Function to wait for server to be ready
wait_for_server() {
    local port=$1
    local timeout_seconds=$TIMEOUT_SECONDS
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
    export TIMEOUT_SECONDS
    export PREFILL_GPUS
    export DECODE_GPUS
    export PREFILL_PORTS
    export DECODE_PORTS
    export PROXY_SERVICE_DISCOVERY_PORT
    export PROXY_APP_PORT
    export MODEL_PATH
    export PROXY_SCRIPT
    
    # Start serve script in background
    "$SCRIPT_DIR/serve.sh" &
    SERVE_PID=$!
    
    # Wait for all servers to start
    echo "Waiting for all servers to start..."
    
    # Parse port arrays
    IFS=',' read -ra PREFILL_PORT_ARRAY <<< "$PREFILL_PORTS"
    IFS=',' read -ra DECODE_PORT_ARRAY <<< "$DECODE_PORTS"
    
    # Wait for prefill servers
    for port in "${PREFILL_PORT_ARRAY[@]}"; do
        if ! wait_for_server $port; then
            echo "Failed to start prefill server on port $port"
            cleanup
            exit 1
        fi
    done
    
    # Wait for decode servers
    for port in "${DECODE_PORT_ARRAY[@]}"; do
        if ! wait_for_server $port; then
            echo "Failed to start decode server on port $port"
            cleanup
            exit 1
        fi
    done
    
    echo "All servers started successfully."
}

# Function to run benchmarks
run_benchmarks() {
    echo "Running benchmarks..."
    
    # Export environment variables for bench script
    export REQUEST_RATES
    export SERVER_PORT=$PROXY_APP_PORT
    export MODEL_PATH
    
    # Start bench script
    "$SCRIPT_DIR/$BENCH_SCRIPT"
    
    echo "Benchmarks completed."
}

# Main execution
main() {
    echo "Starting main control script for P2P NCCL configuration..."
    
    # Check dependencies
    check_dependencies
    
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