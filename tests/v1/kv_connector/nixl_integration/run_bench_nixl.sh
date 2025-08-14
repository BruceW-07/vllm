#!/bin/bash
set -xe

# Get dataset name from command line argument or use default
DATASET_NAME=${1:-"sharegpt"}

# Models to run
MODELS=(
    "/workspace/models/Qwen3-0.6B"
)

# Number of prefill and decode instances to create
NUM_PREFILL_INSTANCES=${NUM_PREFILL_INSTANCES:-1} # Default to 1
NUM_DECODE_INSTANCES=${NUM_DECODE_INSTANCES:-1}   # Default to 1
PREFILLER_TP_SIZE=${PREFILLER_TP_SIZE:-1}
DECODER_TP_SIZE=${DECODER_TP_SIZE:-1}

# Find the git repository root directory
GIT_ROOT=$(git rev-parse --show-toplevel)

SMI_BIN=$(which nvidia-smi || which rocm-smi)

# Benchmark configuration
NUM_PROMPT="100"
REQUEST_RATES=(0.5 1.0 1.5 2.0 2.5 3.0)

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
  echo "  Prompts per test: $NUM_PROMPT"
  echo "  Request rates: ${REQUEST_RATES[*]}"
  echo "================================"

  # Get model-specific arguments
  local model_args=$(get_model_args "$model_name")

  # Arrays to store all hosts and ports
  PREFILL_HOSTS=()
  PREFILL_PORTS=()
  DECODE_HOSTS=()
  DECODE_PORTS=()

  # Start prefill instances
  for i in $(seq 0 $((NUM_PREFILL_INSTANCES-1))); do
    # Calculate GPU ID - we'll distribute across available GPUs
    GPU_ID=$((i + 4 % $(get_num_gpus)))

    # Calculate port number (base port + instance number)
    PORT=$((8150 + i))
    # Calculate side channel port. Avoid clash with with TP workers. 
    SIDE_CHANNEL_PORT=$((5559 + i))

    echo "Starting prefill instance $i on GPU $GPU_ID, port $PORT"

    # Build the command with or without model-specific args
    BASE_CMD="CUDA_VISIBLE_DEVICES=$GPU_ID VLLM_NIXL_SIDE_CHANNEL_PORT=$SIDE_CHANNEL_PORT vllm serve $model_name \
    --port $PORT \
    --enforce-eager \
    --gpu-memory-utilization 0.2 \
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
    # Calculate GPU ID - we'll distribute across available GPUs, starting from after prefill GPUs
    GPU_ID=$(((i + 4 + NUM_PREFILL_INSTANCES) % $(get_num_gpus)))
    # Calculate port number (base port + instance number)
    PORT=$((8250 + i))
    # Calculate side channel port
    SIDE_CHANNEL_PORT=$((5659 + i * $DECODER_TP_SIZE))

    echo "Starting decode instance $i on GPU $GPU_ID, port $PORT"

    # Build the command with or without model-specific args
    BASE_CMD="CUDA_VISIBLE_DEVICES=$GPU_ID VLLM_NIXL_SIDE_CHANNEL_PORT=$SIDE_CHANNEL_PORT vllm serve $model_name \
    --port $PORT \
    --enforce-eager \
    --gpu-memory-utilization 0.2 \
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

  # Start the proxy server
  echo "Starting proxy server with command: $PROXY_CMD"
  $PROXY_CMD &

  # Wait for the proxy to start
  sleep 5

  SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
  
  # Create timestamped results directory to avoid conflicts
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  RESULTS_BASE_DIR="$SCRIPT_DIR/results/$DATASET_NAME"
  RESULTS_DIR="$RESULTS_BASE_DIR/$TIMESTAMP"
  mkdir -p "$RESULTS_DIR"
  
  echo "Results will be saved to: $RESULTS_DIR"

  # Run bench test for each request rate
  for REQUEST_RATE in "${REQUEST_RATES[@]}"; do
    TS=$(date +%Y%m%d_%H%M%S)
    echo "Running vllm bench serve for $model_name with request rate $REQUEST_RATE"
    
    # Create a configuration suffix for the result filename
    CONFIG_SUFFIX="pf${NUM_PREFILL_INSTANCES}d${NUM_DECODE_INSTANCES}tp${PREFILLER_TP_SIZE}x${DECODER_TP_SIZE}"
    
    # Extract model name from model path (last part after /)
    MODEL_NAME=$(basename "$model_name")
    
    vllm bench serve \
      --backend vllm \
      --model "$model_name" \
      --endpoint /v1/completions \
      --dataset-name custom  \
      --dataset-path /workspace/w50052772/data/$DATASET_NAME.jsonl \
      --num-prompt $NUM_PROMPT \
      --request-rate $REQUEST_RATE \
      --seed 42 \
      --port $PROXY_PORT \
      --result-dir $RESULTS_DIR \
      --save-detailed \
      --save-result \
      --metadata \
        "num_prefill_instances=$NUM_PREFILL_INSTANCES" \
        "num_decode_instances=$NUM_DECODE_INSTANCES" \
        "prefiller_tp_size=$PREFILLER_TP_SIZE" \
        "decoder_tp_size=$DECODER_TP_SIZE" \
        "dataset_name=$DATASET_NAME" \
        "model_name=$MODEL_NAME" \
        "config_suffix=$CONFIG_SUFFIX" \
        "gpu_memory_utilization=0.2" \
        "kv_connector=NixlConnector" \
        "prefix_caching=disabled"
    
    echo "Completed benchmark with request rate $REQUEST_RATE"
    
    # Wait a bit between different request rates to let the system stabilize
    sleep 10
  done

  # Clean up before running next model
  cleanup_instances
  sleep 3
}

# Run tests for each model
for idx in "${!MODELS[@]}"; do
  model_name="${MODELS[$idx]}"
  run_tests_for_model "$model_name"
done

echo "All tests completed!"