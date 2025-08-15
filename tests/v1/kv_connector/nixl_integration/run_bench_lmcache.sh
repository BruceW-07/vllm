#!/bin/bash
set -xe

# LMCache vLLM benchmark script using v0 disaggregated prefill-decode separation
# Usage: ./run_bench_lmcache_v0.sh [dataset_name]

# Get dataset name from command line argument or use default
DATASET_NAME=${1:-"sharegpt"}

# Models to run
MODELS=(
    "/workspace/models/Qwen3-0.6B"
)

# LMCache configuration (PD separation using v0 method)
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.8}  # GPU memory utilization
PREFILLER_GPU=${PREFILLER_GPU:-0}  # GPU for prefiller
DECODER_GPU=${DECODER_GPU:-1}      # GPU for decoder

# LMCache server configuration
LMCACHE_SERVER_PORT=${LMCACHE_SERVER_PORT:-8100}

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

# LMCache-related environment variables (v0 setup)
setup_lmcache_env() {
  # Use experimental features in LMCache
  export LMCACHE_USE_EXPERIMENTAL="True"
  # LMCache is set to use 256 tokens per chunk
  export LMCACHE_CHUNK_SIZE="256"
  # Disable local CPU backend in LMCache
  export LMCACHE_LOCAL_CPU="False"
  # Set local CPU memory buffer limit to 5.0 GB
  export LMCACHE_MAX_LOCAL_CPU_SIZE="5.0"
  # Set the remote URL for LMCache server
  export LMCACHE_REMOTE_URL="lm://localhost:${LMCACHE_SERVER_PORT}"
  # Set the serializer/deserializer between vllm and LMCache server
  # `naive` indicates using raw bytes of the tensor without compression
  export LMCACHE_REMOTE_SERDE="naive"
}

# Function to clean up previous instances
cleanup_instances() {
  echo "Cleaning up any running vLLM instances and LMCache server..."
  pkill -f "vllm serve" || true
  pkill -f "lmcache.experimental.server" || true
  pkill -f "toy_proxy_server.py" || true
  sleep 2
}

# Function to start LMCache server
start_lmcache_server() {
  echo "Starting LMCache server on port $LMCACHE_SERVER_PORT"
  python3 -m lmcache.experimental.server localhost $LMCACHE_SERVER_PORT &
  LMCACHE_SERVER_PID=$!
  
  # Wait a bit for server to start
  sleep 3
  
  return $LMCACHE_SERVER_PID
}

# Function to start prefiller instance
start_prefiller() {
  local model_name="$1"
  local model_args="$2"
  
  echo "Starting prefiller on GPU $PREFILLER_GPU, port 8200"
  
  # Set up environment for prefiller
  CUDA_VISIBLE_DEVICES=$PREFILLER_GPU \
    VLLM_USE_V1=0 \
    vllm serve "$model_name" \
    --port 8200 \
    --disable-log-requests \
    --enforce-eager \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --disable-frontend-multiprocessing \
    --no-enable-prefix-caching \
    --kv-transfer-config \
    '{"kv_connector":"LMCacheConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2}' \
    $model_args &
  
  PREFILLER_PID=$!
}

# Function to start decoder instance  
start_decoder() {
  local model_name="$1"
  local model_args="$2"
  
  echo "Starting decoder on GPU $DECODER_GPU, port 8300"
  
  # Set up environment for decoder
  CUDA_VISIBLE_DEVICES=$DECODER_GPU \
    VLLM_USE_V1=0 \
    vllm serve "$model_name" \
    --port 8300 \
    --disable-log-requests \
    --enforce-eager \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --disable-frontend-multiprocessing \
    --no-enable-prefix-caching \
    --kv-transfer-config \
    '{"kv_connector":"LMCacheConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2}' \
    $model_args &
  
  DECODER_PID=$!
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

# Function to run tests for a specific model using v0 LMCache disagg approach
run_tests_for_model() {
  local model_name="$1"
  echo "================================"
  echo "Testing model: $model_name (LMCache v0 Mode)"
  echo "Configuration:"
  echo "  Dataset: $DATASET_NAME"
  echo "  GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
  echo "  Prefiller GPU: $PREFILLER_GPU"
  echo "  Decoder GPU: $DECODER_GPU"
  echo "  Prompts per test: $NUM_PROMPT"
  echo "  Request rates: ${REQUEST_RATES[*]}"
  echo "  LMCache Server Port: $LMCACHE_SERVER_PORT"
  echo "================================"

  # Validate GPU allocation
  local total_gpus=$(get_num_gpus)
  local max_gpu=$(( PREFILLER_GPU > DECODER_GPU ? PREFILLER_GPU : DECODER_GPU ))
  
  echo "GPU Allocation Validation:"
  echo "  Total available GPUs: $total_gpus"
  echo "  Prefiller GPU: $PREFILLER_GPU"
  echo "  Decoder GPU: $DECODER_GPU"
  echo "  Highest GPU ID to be used: $max_gpu"

  if [ "$max_gpu" -ge "$total_gpus" ]; then
    echo "ERROR: Not enough GPUs available! Need GPU IDs up to $max_gpu, but only have GPUs 0 to $((total_gpus - 1))"
    exit 1
  fi

  echo "  ✓ GPU allocation is valid"
  echo ""

  # Get model-specific arguments
  local model_args=$(get_model_args "$model_name")

  # Clean up any existing instances
  cleanup_instances

  # Set up LMCache environment variables
  setup_lmcache_env

  # Start LMCache server
  start_lmcache_server
  LMCACHE_SERVER_PID=$!

  # Start prefiller and decoder
  start_prefiller "$model_name" "$model_args"
  start_decoder "$model_name" "$model_args"
  
  # Wait for instances to start
  echo "Waiting for prefiller to start..."
  wait_for_server 8200
  echo "Waiting for decoder to start..."
  wait_for_server 8300

  # Start proxy server to route requests
  PROXY_PORT=8400
  echo "Starting proxy server on port $PROXY_PORT"
  python3 "${GIT_ROOT}/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py" \
    --port $PROXY_PORT \
    --prefiller-hosts localhost \
    --prefiller-ports 8200 \
    --decoder-hosts localhost \
    --decoder-ports 8300 &
  PROXY_PID=$!

  # Wait for proxy to start
  echo "Waiting for proxy server to start..."
  wait_for_server $PROXY_PORT

  echo "All LMCache components are ready. Starting benchmark tests..."

  # Create results directory with timestamp
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  RESULTS_DIR="${GIT_ROOT}/tests/v1/kv_connector/nixl_integration/results_lmcache_${TIMESTAMP}"
  mkdir -p "$RESULTS_DIR"

  echo "Results will be saved to: $RESULTS_DIR"

  # Extract model name from model path (last part after /)
  MODEL_NAME=$(basename "$model_name")
  
  # Create a configuration suffix for the result filename
  CONFIG_SUFFIX="lmcache_v0_pf${PREFILLER_GPU}d${DECODER_GPU}"

  # Run bench test for each request rate
  for REQUEST_RATE in "${REQUEST_RATES[@]}"; do
    echo "Running vllm bench serve for $model_name with request rate $REQUEST_RATE"
    
    vllm bench serve \
      --backend vllm \
      --model "$model_name" \
      --endpoint /v1/completions \
      --dataset-name custom \
      --dataset-path /workspace/w50052772/data/$DATASET_NAME.jsonl \
      --num-prompt $NUM_PROMPT \
      --request-rate $REQUEST_RATE \
      --seed 42 \
      --port $PROXY_PORT \
      --result-dir $RESULTS_DIR \
      --save-detailed \
      --save-result \
      --metadata \
        "prefiller_gpu=$PREFILLER_GPU" \
        "decoder_gpu=$DECODER_GPU" \
        "gpu_memory_utilization=$GPU_MEMORY_UTILIZATION" \
        "dataset_name=$DATASET_NAME" \
        "model_name=$MODEL_NAME" \
        "config_suffix=$CONFIG_SUFFIX" \
        "deployment_mode=lmcache_v0" \
        "kv_connector=LMCacheConnector" \
        "prefix_caching=disabled" \
        "lmcache_server_port=$LMCACHE_SERVER_PORT"
    
    echo "Completed benchmark with request rate $REQUEST_RATE"
    
    # Wait a bit between different request rates to let the system stabilize
    sleep 10
  done

  echo "All benchmark tests completed for model: $model_name"
  echo "Results saved in: $RESULTS_DIR"
  
  # Clean up
  cleanup_instances
}

# Main execution
echo "Starting LMCache v0 benchmark tests..."

# Check dependencies
echo "Checking LMCache dependencies..."
python3 -c "import lmcache" || { echo "LMCache not installed. Please install it first."; exit 1; }

# Clean up any existing instances before starting
cleanup_instances

# Run tests for each model
for model in "${MODELS[@]}"; do
  run_tests_for_model "$model"
done

echo "All LMCache v0 benchmark tests completed!"
