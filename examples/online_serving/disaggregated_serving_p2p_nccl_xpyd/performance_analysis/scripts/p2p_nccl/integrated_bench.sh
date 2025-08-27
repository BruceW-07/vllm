#!/bin/bash

# =============================================================================
# Integrated Benchmark Script for vLLM Disaggregated Serving
# =============================================================================

set -xe  # Exit on any error

# Configuration - can be overridden via environment variables
MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}
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
DECODE_LOG="decode1.log"
RESULTS_DIR="results"
REQUEST_RATES=(1 2 3 4 5 6 7 8 9 10 11)

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
        kill -9 "${PIDS[@]}" 2>/dev/null || true
        wait "${PIDS[@]}" 2>/dev/null || true
    fi
    exit 0
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

# Function to clear decode log
clear_decode_log() {
    if [[ -f "$DECODE_LOG" ]]; then
        echo "Clearing $DECODE_LOG..."
        > "$DECODE_LOG"  # Truncate the file
    else
        echo "Creating $DECODE_LOG..."
        touch "$DECODE_LOG"
    fi
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
    echo "  Model: $MODEL"
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
        CUDA_VISIBLE_DEVICES=$gpu_id VLLM_USE_V1=1 vllm serve $MODEL \
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
        "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_producer\",\"kv_buffer_size\":\"1e1\",\"kv_port\":\"$kv_port\",\"kv_connector_extra_config\":{\"proxy_ip\":\"0.0.0.0\",\"proxy_port\":\"$PROXY_SERVICE_DISCOVERY_PORT\",\"http_port\":\"$port\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"16\"}}" > prefill$((i+1)).log 2>&1 &
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
        VLLM_USE_V1=1 CUDA_VISIBLE_DEVICES=$gpu_id vllm serve $MODEL \
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
        "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_consumer\",\"kv_buffer_size\":\"8e9\",\"kv_port\":\"$kv_port\",\"kv_connector_extra_config\":{\"proxy_ip\":\"0.0.0.0\",\"proxy_port\":\"$PROXY_SERVICE_DISCOVERY_PORT\",\"http_port\":\"$port\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"16\"}}" > decode$((i+1)).log 2>&1 &
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
    echo "Starting integrated benchmark script..."
    
    # Create results directory
    mkdir -p "$RESULTS_DIR"
    
    # Start the proxy server
    echo "Starting proxy server..."
    start_proxy
    
    # Start the serving components
    echo "Starting serving components..."
    start_serving
    
    # Run benchmarks
    for REQUEST_RATE in "${REQUEST_RATES[@]}"; do
        NUM_PROMPTS=$((REQUEST_RATE * 300))
        echo "Running benchmark with request rate: $REQUEST_RATE, total prompts: $NUM_PROMPTS"
        
        # Clear the decode log before each benchmark run
        clear_decode_log
        
        # Run the benchmark
        vllm bench serve \
            --backend vllm \
            --model "$MODEL_PATH" \
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
            --port $PROXY_APP_PORT
            
        echo "Benchmark with request rate $REQUEST_RATE and $NUM_PROMPTS prompts completed."
        sleep 10
    done
    
    echo "All benchmarks completed."
    
    # Cleanup
    cleanup
}

# Run main function
main "$@"