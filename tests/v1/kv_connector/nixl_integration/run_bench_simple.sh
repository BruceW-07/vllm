#!/bin/bash
set -xe

# Simple vLLM benchmark script without prefill-decode separation
# Usage: ./run_bench_simple.sh [dataset_name]

# Get dataset name from command line argument or use default
DATASET_NAME=${1:-"sharegpt"}

# Models to run
MODELS=(
    "/workspace/models/Qwen3-0.6B"
)

# Simple configuration (no PD separation)
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.8}

# Find the git repository root directory
GIT_ROOT=$(git rev-parse --show-toplevel)

SMI_BIN=$(which nvidia-smi || which rocm-smi)

# Benchmark configuration
NUM_PROMPT="100"
REQUEST_RATES=(0.5 1.0 1.5 2.0 2.5 3.0)

# Trap the SIGINT signal (triggered by Ctrl+C)
trap 'pkill -f "vllm serve" || true; exit' SIGINT SIGTERM EXIT

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
  echo "Testing model: $model_name (Simple Mode)"
  echo "Configuration:"
  echo "  Dataset: $DATASET_NAME"
  echo "  Tensor Parallel size: $TENSOR_PARALLEL_SIZE"
  echo "  GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
  echo "  Prompts per test: $NUM_PROMPT"
  echo "  Request rates: ${REQUEST_RATES[*]}"
  echo "================================"

  # Get model-specific arguments
  local model_args=$(get_model_args "$model_name")

  # Use first available GPU
  GPU_ID=0
  SERVER_PORT=8000

  echo "Starting vLLM server on GPU $GPU_ID, port $SERVER_PORT"

  # Build the command with or without model-specific args
  BASE_CMD="CUDA_VISIBLE_DEVICES=$GPU_ID vllm serve $model_name \
    --port $SERVER_PORT \
    --enforce-eager \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE"

  if [ -n "$model_args" ]; then
    FULL_CMD="$BASE_CMD $model_args"
  else
    FULL_CMD="$BASE_CMD"
  fi

  # Start vLLM server in background
  eval "$FULL_CMD &"
  
  # Store the server PID
  SERVER_PID=$!

  # Wait for server to start
  echo "Waiting for vLLM server to start..."
  wait_for_server $SERVER_PORT

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
    CONFIG_SUFFIX="simple_tp${TENSOR_PARALLEL_SIZE}"
    
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
      --port $SERVER_PORT \
      --result-dir $RESULTS_DIR \
      --save-detailed \
      --save-result \
      --metadata \
        "tensor_parallel_size=$TENSOR_PARALLEL_SIZE" \
        "gpu_memory_utilization=$GPU_MEMORY_UTILIZATION" \
        "dataset_name=$DATASET_NAME" \
        "model_name=$MODEL_NAME" \
        "config_suffix=$CONFIG_SUFFIX" \
        "deployment_mode=simple" \
        "prefix_caching=enabled"
    
    echo "Completed benchmark with request rate $REQUEST_RATE"
    
    # Wait a bit between different request rates to let the system stabilize
    sleep 5
  done

  # Clean up server
  echo "Stopping vLLM server (PID: $SERVER_PID)"
  kill $SERVER_PID 2>/dev/null || true
  sleep 3
}

# Clean up any existing instances before starting
cleanup_instances

# Run tests for each model
for idx in "${!MODELS[@]}"; do
  model_name="${MODELS[$idx]}"
  run_tests_for_model "$model_name"
done

echo "All tests completed!"
