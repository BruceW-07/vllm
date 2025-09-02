#!/bin/bash

# =============================================================================
# Serve Script for vLLM Disaggregated Serving with P2P NCCL
# =============================================================================

set -xe  # Exit on any error

# Benchmark configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default model path - update this to your actual model path
MODEL_PATH="/workspace/models/Llama-3.1-8B-Instruct"

# Global variables
PIDS=()

# Cleanup function
cleanup() {
    echo "Stopping everything…"
    trap - INT TERM        # prevent re-entrancy
    
    # Kill all background processes
    if [[ ${#PIDS[@]} -gt 0 ]]; then
        echo "Killing ${#PIDS[@]} background processes..."
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                echo "Killing process $pid"
                kill -TERM "$pid" 2>/dev/null || true
            fi
        done
        
        # Wait for processes to terminate gracefully
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                echo "Waiting for process $pid to terminate..."
                timeout 10 wait "$pid" 2>/dev/null || true
            fi
        done
        
        # Force kill any remaining processes
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                echo "Force killing process $pid"
                kill -KILL "$pid" 2>/dev/null || true
            fi
        done
    fi
    
    # Check if any processes are still running after PID-based cleanup
    echo "Verifying all processes have been terminated..."
    
    local remaining_processes=0
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Warning: Process $pid is still running after cleanup"
            remaining_processes=$((remaining_processes + 1))
        fi
    done
    
    if [[ $remaining_processes -gt 0 ]]; then
        echo "Warning: $remaining_processes processes are still running"
        echo "You may need to manually terminate them if needed"
    else
        echo "All processes terminated successfully"
    fi
    
    # Wait a bit for processes to fully terminate
    sleep 2
    
    echo "Cleanup completed."
}

# Trap signals to ensure cleanup
trap cleanup INT TERM EXIT

# Function to check number of GPUs
check_num_gpus() {
    # Check if the number of GPUs are >=2 via nvidia-smi
    num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    if [ "$num_gpus" -lt 2 ]; then
        echo "You need at least 2 GPUs to run disaggregated prefill."
        exit 1
    else
        echo "Found $num_gpus GPUs."
    fi
}

# Function to ensure python libraries are installed
ensure_python_library_installed() {
    echo "Checking if $1 is installed..."
    if ! python3 -c "import $1" > /dev/null 2>&1; then
        echo "$1 is not installed. Please install it via pip install $1."
        exit 1
    else
        echo "$1 is installed."
    fi
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

# Function to start proxy server
start_proxy() {
    echo "Starting proxy server..."
    
    # Check if proxy script exists
    if [[ ! -f "$PROXY_SCRIPT" ]]; then
        echo "Proxy script not found: $PROXY_SCRIPT"
        exit 1
    fi
    
    # Start proxy server with specified ports
    python3 "$PROXY_SCRIPT" \
        --service-discovery-port "$PROXY_SERVICE_DISCOVERY_PORT" \
        --app-port "$PROXY_APP_PORT" &
    PIDS+=($!)
    
    # Wait a bit for proxy to start
    sleep 5
    
    echo "Proxy server started."
}

# Function to start serving components
start_serving() {
    echo "Warning: P2P NCCL disaggregated prefill XpYd support for vLLM v1 is experimental and subject to change."
    echo ""
    echo "Architecture Configuration:"
    echo "  Model: $MODEL_PATH"
    echo "  Prefill GPUs: $PREFILL_GPUS, Ports: $PREFILL_PORTS"
    echo "  Decode GPUs: $DECODE_GPUS, Ports: $DECODE_PORTS"
    echo "  Proxy Service Discovery Port: $PROXY_SERVICE_DISCOVERY_PORT"
    echo "  Proxy App Port: $PROXY_APP_PORT"
    echo "  Timeout: ${TIMEOUT_SECONDS}s"
    echo ""
    
    # Check system requirements
    check_num_gpus
    ensure_python_library_installed pandas
    ensure_python_library_installed datasets
    ensure_python_library_installed vllm
    ensure_python_library_installed quart
    
    echo "Launching disaggregated serving components..."
    echo "Please check the log files for detailed output:"
    echo "  - prefill*.log: Prefill server logs"
    echo "  - decode*.log: Decode server logs"
    echo "  - proxy.log: Proxy server log"
    
    # Parse GPU and port arrays
    IFS=',' read -ra PREFILL_GPU_ARRAY <<< "$PREFILL_GPUS"
    IFS=',' read -ra DECODE_GPU_ARRAY <<< "$DECODE_GPUS"
    IFS=',' read -ra PREFILL_PORT_ARRAY <<< "$PREFILL_PORTS"
    IFS=',' read -ra DECODE_PORT_ARRAY <<< "$DECODE_PORTS"
    
    # =============================================================================
    # Launch Prefill Servers (X Producers)
    # =============================================================================
    echo ""
    echo "Starting ${#PREFILL_GPU_ARRAY[@]} prefill server(s)..."
    for i in "${!PREFILL_GPU_ARRAY[@]}"; do
        local gpu_id=${PREFILL_GPU_ARRAY[$i]}
        local port=${PREFILL_PORT_ARRAY[$i]}
        local kv_port=$((21001 + i))
        
        echo "  Prefill server $((i+1)): GPU $gpu_id, Port $port, KV Port $kv_port"
        CUDA_VISIBLE_DEVICES=$gpu_id VLLM_USE_V1=1 vllm serve $MODEL_PATH \
        --no-enable-prefix-caching \
        --host 0.0.0.0 \
        --port $port \
        --tensor-parallel-size 1 \
        --seed 1024 \
        --dtype float16 \
        --max-model-len 10000 \
        --max-num-batched-tokens 10000 \
        --max-num-seqs 256 \
        --trust-remote-code \
        --gpu-memory-utilization 0.9 \
        --kv-transfer-config \
        "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_producer\",\"kv_buffer_size\":\"1e1\",\"kv_port\":\"$kv_port\",\"kv_connector_extra_config\":{\"proxy_ip\":\"0.0.0.0\",\"proxy_port\":\"$PROXY_SERVICE_DISCOVERY_PORT\",\"http_port\":\"$port\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"16\"}}" > "$SCRIPT_DIR/prefill$((i+1)).log" 2>&1 &
        PIDS+=($!)
    done
    
    # =============================================================================
    # Launch Decode Servers (Y Decoders)
    # =============================================================================
    echo ""
    echo "Starting ${#DECODE_GPU_ARRAY[@]} decode server(s)..."
    for i in "${!DECODE_GPU_ARRAY[@]}"; do
        local gpu_id=${DECODE_GPU_ARRAY[$i]}
        local port=${DECODE_PORT_ARRAY[$i]}
        local kv_port=$((22001 + i))
        
        echo "  Decode server $((i+1)): GPU $gpu_id, Port $port, KV Port $kv_port"
        VLLM_USE_V1=1 CUDA_VISIBLE_DEVICES=$gpu_id vllm serve $MODEL_PATH \
        --no-enable-prefix-caching \
        --host 0.0.0.0 \
        --port $port \
        --tensor-parallel-size 1 \
        --seed 1024 \
        --dtype float16 \
        --max-model-len 10000 \
        --max-num-batched-tokens 10000 \
        --max-num-seqs 256 \
        --trust-remote-code \
        --gpu-memory-utilization 0.7 \
        --kv-transfer-config \
        "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_consumer\",\"kv_buffer_size\":\"8e9\",\"kv_port\":\"$kv_port\",\"kv_connector_extra_config\":{\"proxy_ip\":\"0.0.0.0\",\"proxy_port\":\"$PROXY_SERVICE_DISCOVERY_PORT\",\"http_port\":\"$port\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"16\"}}" > "$SCRIPT_DIR/decode$((i+1)).log" 2>&1 &
        PIDS+=($!)
    done
    
    # =============================================================================
    # Wait for All Servers to Start
    # =============================================================================
    echo ""
    echo "Waiting for all servers to start..."
    for port in "${PREFILL_PORT_ARRAY[@]}" "${DECODE_PORT_ARRAY[@]}"; do
        if ! wait_for_server $port; then
            echo "Failed to start server on port $port"
            cleanup
            exit 1
        fi
    done
    
    echo "All servers started successfully."
}

# Main execution
main() {
    echo "Starting serve script for P2P NCCL configuration..."
    
    # Start the proxy server
    echo "Starting proxy server..."
    start_proxy
    
    # Start the serving components
    echo "Starting serving components..."
    start_serving
    
    # Keep the script running
    echo "Serve components are running. Press Ctrl+C to stop."
    wait
}

# Run main function
main "$@"