#!/bin/bash
set -e
set +x

# Get dataset name from command line argument or use default
DATASET_NAME=${1:-"sharegpt"}

# Models to run
MODELS=(
    "/workspace/models/Qwen3-8B"
)

# Number of prefill and decode instances to create
NUM_PREFILL_INSTANCES=${NUM_PREFILL_INSTANCES:-1} # Default to 1
NUM_DECODE_INSTANCES=${NUM_DECODE_INSTANCES:-1}   # Default to 1
PREFILLER_TP_SIZE=${PREFILLER_TP_SIZE:-1}
DECODER_TP_SIZE=${DECODER_TP_SIZE:-1}

# GPU allocation and memory settings
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.8}  # GPU memory utilization
START_GPU_ID=${START_GPU_ID:-0}  # Starting GPU ID to use

# Find the git repository root directory
GIT_ROOT=$(git rev-parse --show-toplevel)

SMI_BIN=$(which nvidia-smi || which rocm-smi)

# Trap the SIGINT signal (triggered by Ctrl+C)
trap 'kill $(jobs -pr)' SIGINT SIGTERM EXIT

# Waits for vLLM to start.
wait_for_server() {
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

# Function to clean up previous instances
cleanup_instances() {
  echo "Cleaning up any running vLLM instances..."
  pkill -f "vllm serve" || true
  sleep 2
}

# Handle to get model-specific arguments for deepseek
get_model_args() {
  local model_name=$1
  local extra_args=""

  if [[ "$model_name" == "deepseek-ai/deepseek-vl2-tiny" ]]; then
    extra_args="--hf_overrides '{\"architectures\": [\"DeepseekVLV2ForCausalLM\"]}' --trust-remote-code"
  fi

  echo "$extra_args"
}

get_num_gpus() {
  if [[ "$SMI_BIN" == *"nvidia"* ]]; then
    echo "$($SMI_BIN --query-gpu=name --format=csv,noheader | wc -l)"
  else
    echo "$($SMI_BIN -l | grep GPU | wc -l)"
  fi
}

# Function to run tests for a specific model
run_tests_for_model() {
  local model_name="$1"
  echo "================================"
  echo "Testing model: $model_name"
  echo "Configuration:"
  echo "  Dataset: $DATASET_NAME"
  echo "  Prefill instances: $NUM_PREFILL_INSTANCES"
  echo "  Decode instances: $NUM_DECODE_INSTANCES"
  echo "  Prefiller TP size: $PREFILLER_TP_SIZE"
  echo "  Decoder TP size: $DECODER_TP_SIZE"
  echo "  GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
  echo "  Starting GPU ID: $START_GPU_ID"
  echo "  Prompts per test: $NUM_PROMPT"
  echo "  Request rates: ${REQUEST_RATES[*]}"
  echo "================================"

  # Display GPU allocation strategy
  echo "GPU Allocation Strategy:"
  echo "  Total available GPUs: $(get_num_gpus)"
  echo "  Starting GPU ID: $START_GPU_ID"
  echo "  Prefill instances will use GPUs starting from: $START_GPU_ID"
  echo "  Decode instances will use GPUs starting from: $((START_GPU_ID + NUM_PREFILL_INSTANCES))"
  echo ""

  # Get model-specific arguments
  local model_args=$(get_model_args "$model_name")

  # Arrays to store all hosts and ports
  PREFILL_HOSTS=()
  PREFILL_PORTS=()
  DECODE_HOSTS=()
  DECODE_PORTS=()

  # Start prefill instances
  for i in $(seq 0 $((NUM_PREFILL_INSTANCES-1))); do
    # Calculate GPU ID - distribute across available GPUs starting from START_GPU_ID
    if [ "$PREFILLER_TP_SIZE" -eq 1 ]; then
      GPU_ID=$(((START_GPU_ID + i) % $(get_num_gpus)))
      GPU_DEVICES="$GPU_ID"
    else
      # For TP > 1, use consecutive GPUs starting from calculated base
      base_gpu=$(((START_GPU_ID + i) % $(get_num_gpus)))
      GPU_DEVICES=$(seq -s, $base_gpu $((base_gpu + PREFILLER_TP_SIZE - 1)))
    fi

    # Calculate port number (base port + instance number)
    PORT=$((8157 + i))
    # Calculate side channel port. Avoid clash with TP workers. 
    SIDE_CHANNEL_PORT=$((5564 + i))

    echo "Starting prefill instance $i on GPU(s) $GPU_DEVICES, port $PORT (TP size: $PREFILLER_TP_SIZE)"

    # Build the command with or without model-specific args
    BASE_CMD="CUDA_VISIBLE_DEVICES=$GPU_DEVICES VLLM_NIXL_SIDE_CHANNEL_PORT=$SIDE_CHANNEL_PORT vllm serve $model_name \
    --port $PORT \
    --enforce-eager \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --tensor-parallel-size $PREFILLER_TP_SIZE \
    --kv-transfer-config '{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\"}' \
    --no-enable-prefix-caching"

    if [ -n "$model_args" ]; then
    FULL_CMD="$BASE_CMD $model_args"
    else
    FULL_CMD="$BASE_CMD"
    fi

    eval "$FULL_CMD &"

    # Store host and port for proxy configuration
    PREFILL_HOSTS+=("localhost")
    PREFILL_PORTS+=($PORT)
  done

  # Start decode instances
  for i in $(seq 0 $((NUM_DECODE_INSTANCES-1))); do
    # Calculate GPU ID - distribute across available GPUs, starting from after prefill instances
    if [ "$DECODER_TP_SIZE" -eq 1 ]; then
      GPU_ID=$(((START_GPU_ID + i + NUM_PREFILL_INSTANCES) % $(get_num_gpus)))
      GPU_DEVICES="$GPU_ID"
    else
      # For TP > 1, use consecutive GPUs starting from calculated base
      base_gpu=$(((START_GPU_ID + i + NUM_PREFILL_INSTANCES) % $(get_num_gpus)))
      GPU_DEVICES=$(seq -s, $base_gpu $((base_gpu + DECODER_TP_SIZE - 1)))
    fi
    
    # Calculate port number (base port + instance number)
    PORT=$((8257 + i))
    # Calculate side channel port
    SIDE_CHANNEL_PORT=$((5664 + i * $DECODER_TP_SIZE))

    echo "Starting decode instance $i on GPU(s) $GPU_DEVICES, port $PORT (TP size: $DECODER_TP_SIZE)"

    # Build the command with or without model-specific args
    BASE_CMD="CUDA_VISIBLE_DEVICES=$GPU_DEVICES VLLM_NIXL_SIDE_CHANNEL_PORT=$SIDE_CHANNEL_PORT vllm serve $model_name \
    --port $PORT \
    --enforce-eager \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --tensor-parallel-size $DECODER_TP_SIZE \
    --kv-transfer-config '{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\"}' \
    --no-enable-prefix-caching"

    if [ -n "$model_args" ]; then
    FULL_CMD="$BASE_CMD $model_args"
    else
    FULL_CMD="$BASE_CMD"
    fi

    eval "$FULL_CMD &"

    # Store host and port for proxy configuration
    DECODE_HOSTS+=("localhost")
    DECODE_PORTS+=($PORT)
  done

  # Wait for all instances to start
  for PORT in "${PREFILL_PORTS[@]}"; do
    echo "Waiting for prefill instance on port $PORT to start..."
    wait_for_server $PORT
  done

  for PORT in "${DECODE_PORTS[@]}"; do
    echo "Waiting for decode instance on port $PORT to start..."
    wait_for_server $PORT
  done

  # Build the command for the proxy server with all the hosts and ports
  PROXY_PORT=8077
  PROXY_CMD="python ${GIT_ROOT}/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py --port ${PROXY_PORT}"

  # Add all prefill hosts and ports
  PROXY_CMD+=" --prefiller-hosts ${PREFILL_HOSTS[@]}"
  PROXY_CMD+=" --prefiller-ports ${PREFILL_PORTS[@]}"

  # Add all decode hosts and ports
  PROXY_CMD+=" --decoder-hosts ${DECODE_HOSTS[@]}"
  PROXY_CMD+=" --decoder-ports ${DECODE_PORTS[@]}"

  # Add TP size parameters (if toy_proxy_server.py supports them)
  # PROXY_CMD+=" --prefiller-tp-size ${PREFILLER_TP_SIZE}"
  # PROXY_CMD+=" --decoder-tp-size ${DECODER_TP_SIZE}"

  # Start the proxy server
  echo "Starting proxy server with command: $PROXY_CMD"
  $PROXY_CMD &

  # Wait for the proxy to start
  echo "Waiting for proxy server on port $PROXY_PORT to start..."
  wait_for_server $PROXY_PORT

  echo "All services are running! Proxy server available at http://localhost:$PROXY_PORT"
  echo "Press Ctrl+C to stop all services..."
  
  # Keep the services running
  while true; do
    sleep 60
  done

}

# Run tests for each model
for idx in "${!MODELS[@]}"; do
  model_name="${MODELS[$idx]}"
  run_tests_for_model "$model_name"
done

echo "All tests completed!"