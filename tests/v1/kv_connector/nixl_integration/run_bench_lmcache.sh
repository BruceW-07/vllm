#!/bin/bash
set -xe

# LMCache vLLM benchmark script using prefill-decode separation
# Usage: ./run_bench_lmcache.sh [dataset_name]

# Get dataset name from command line argument or use default
DATASET_NAME=${1:-"sharegpt"}

# Models to run
MODELS=(
    "/workspace/models/Qwen3-0.6B"
)

# LMCache configuration (with PD separation)
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.2}  # GPU memory utilization
PREFILLER_GPU=${PREFILLER_GPU:-0}  # GPU for prefiller
DECODER_GPU=${DECODER_GPU:-1}      # GPU for decoder

# Find the git repository root directory
GIT_ROOT=$(git rev-parse --show-toplevel)

SMI_BIN=$(which nvidia-smi || which rocm-smi)

# Benchmark configuration
NUM_PROMPT="100"
REQUEST_RATES=(0.5 1.0 1.5 2.0 2.5 3.0)

# LMCache directories and config
LMCACHE_DIR="${GIT_ROOT}/examples/others/lmcache/disagg_prefill_lmcache_v1"

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
  pkill -f "disagg_proxy_server.py" || true
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

# Function to start LMCache prefiller
start_lmcache_prefiller() {
  local model_name="$1"
  local model_args="$2"
  
  echo "Starting LMCache prefiller on GPU $PREFILLER_GPU, port 8100"
  
  prefill_config_file="${LMCACHE_DIR}/configs/lmcache-prefiller-config.yaml"
  
  # Start prefiller
  UCX_TLS=cuda_ipc,cuda_copy,tcp \
    LMCACHE_CONFIG_FILE="$prefill_config_file" \
    LMCACHE_USE_EXPERIMENTAL=True \
    VLLM_ENABLE_V1_MULTIPROCESSING=1 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    CUDA_VISIBLE_DEVICES="$PREFILLER_GPU" \
    vllm serve "$model_name" \
    --port 8100 \
    --disable-log-requests \
    --enforce-eager \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --kv-transfer-config \
    '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_producer","kv_connector_extra_config": {"discard_partial_chunks": false, "lmcache_rpc_port": "producer1"}}' \
    $model_args &
}

# Function to start LMCache decoder
start_lmcache_decoder() {
  local model_name="$1"
  local model_args="$2"
  
  echo "Starting LMCache decoder on GPU $DECODER_GPU, port 8200"
  
  decode_config_file="${LMCACHE_DIR}/configs/lmcache-decoder-config.yaml"
  
  # Start decoder
  UCX_TLS=cuda_ipc,cuda_copy,tcp \
    LMCACHE_CONFIG_FILE="$decode_config_file" \
    LMCACHE_USE_EXPERIMENTAL=True \
    VLLM_ENABLE_V1_MULTIPROCESSING=1 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    CUDA_VISIBLE_DEVICES="$DECODER_GPU" \
    vllm serve "$model_name" \
    --port 8200 \
    --disable-log-requests \
    --enforce-eager \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --kv-transfer-config \
    '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_consumer","kv_connector_extra_config": {"discard_partial_chunks": false, "lmcache_rpc_port": "consumer1"}}' \
    $model_args &
}

# Function to start LMCache proxy server
start_lmcache_proxy() {
  echo "Starting LMCache proxy server on port 9000"
  
  python3 "${LMCACHE_DIR}/disagg_proxy_server.py" \
    --host localhost \
    --port 9000 \
    --prefiller-host localhost \
    --prefiller-port 8100 \
    --decoder-host localhost \
    --decoder-port 8200 &
}

# Function to run tests for a specific model
run_tests_for_model() {
  local model_name="$1"
  echo "================================"
  echo "Testing model: $model_name (LMCache Mode)"
  echo "Configuration:"
  echo "  Dataset: $DATASET_NAME"
  echo "  GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
  echo "  Prefiller GPU: $PREFILLER_GPU"
  echo "  Decoder GPU: $DECODER_GPU"
  echo "  Prompts per test: $NUM_PROMPT"
  echo "  Request rates: ${REQUEST_RATES[*]}"
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

  # Start LMCache components
  start_lmcache_prefiller "$model_name" "$model_args"
  start_lmcache_decoder "$model_name" "$model_args"
  
  # Wait for prefiller and decoder to start
  echo "Waiting for prefiller to start..."
  wait_for_server 8100
  echo "Waiting for decoder to start..."
  wait_for_server 8200
  
  # Start proxy server
  start_lmcache_proxy
  
  # Wait for proxy to start
  echo "Waiting for proxy server to start..."
  wait_for_server 9000

  echo "All LMCache components are ready. Starting benchmark tests..."

  # Create results directory with timestamp
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  RESULTS_DIR="${GIT_ROOT}/tests/v1/kv_connector/nixl_integration/results_lmcache_${TIMESTAMP}"
  mkdir -p "$RESULTS_DIR"

  # Save metadata
  cat > "$RESULTS_DIR/metadata.json" << EOF
{
  "timestamp": "$TIMESTAMP",
  "dataset_name": "$DATASET_NAME",
  "model_name": "$model_name",
  "mode": "lmcache",
  "gpu_memory_utilization": $GPU_MEMORY_UTILIZATION,
  "prefiller_gpu": $PREFILLER_GPU,
  "decoder_gpu": $DECODER_GPU,
  "num_prompts": $NUM_PROMPT,
  "request_rates": [$(IFS=,; echo "${REQUEST_RATES[*]}")],
  "git_commit": "$(cd $GIT_ROOT && git rev-parse HEAD 2>/dev/null || echo 'unknown')",
  "git_repo": "$(cd $GIT_ROOT && git config --get remote.origin.url 2>/dev/null || echo 'unknown')"
}
EOF

  # Run benchmark for each request rate
  for rate in "${REQUEST_RATES[@]}"; do
    echo "Running benchmark with request rate: $rate req/s"
    
    RESULT_FILE="$RESULTS_DIR/vllm-${rate}qps-${TIMESTAMP}.json"
    
    cd "$GIT_ROOT"
    python3 benchmarks/benchmark_serving.py \
      --backend vllm \
      --host localhost \
      --port 9000 \
      --model "$model_name" \
      --dataset-name "$DATASET_NAME" \
      --num-prompts "$NUM_PROMPT" \
      --request-rate "$rate" \
      --save-detailed "$RESULT_FILE" \
      --metadata "$RESULTS_DIR/metadata.json"
    
    echo "Benchmark completed for rate $rate req/s, results saved to $RESULT_FILE"
    
    # Wait a bit between tests
    sleep 2
  done

  echo "All benchmark tests completed for model: $model_name"
  echo "Results saved in: $RESULTS_DIR"
  
  # Clean up
  cleanup_instances
}
  echo "================================"

  # Display GPU allocation strategy
  echo "GPU Allocation Strategy (similar to run_accuracy_test.sh):"
  echo "  Total available GPUs: $(get_num_gpus)"
  echo "  Prefill instances will use GPUs in round-robin: 0, 1, 2, ..., $(($(get_num_gpus) - 1)), 0, 1, ..."
  echo "  Decode instances will start from GPU index: $NUM_PREFILL_INSTANCES"
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
    # Calculate GPU ID - distribute across available GPUs (like run_accuracy_test.sh)
    if [ "$PREFILLER_TP_SIZE" -eq 1 ]; then
      GPU_ID=$((i % $(get_num_gpus)))
      GPU_DEVICES="$GPU_ID"
    else
      # For TP > 1, use consecutive GPUs starting from calculated base
      base_gpu=$((i % $(get_num_gpus)))
      GPU_DEVICES=$(seq -s, $base_gpu $((base_gpu + PREFILLER_TP_SIZE - 1)))
    fi

    # Calculate port number (base port + instance number)
    PORT=$((8100 + i))

    echo "Starting prefill instance $i on GPU(s) $GPU_DEVICES, port $PORT (TP size: $PREFILLER_TP_SIZE)"

    # Build the command with or without model-specific args
    BASE_CMD="UCX_TLS=cuda_ipc,cuda_copy,tcp \
      LMCACHE_USE_EXPERIMENTAL=True \
      VLLM_ENABLE_V1_MULTIPROCESSING=1 \
      VLLM_WORKER_MULTIPROC_METHOD=spawn \
      CUDA_VISIBLE_DEVICES=$GPU_DEVICES \
      vllm serve $model_name \
      --port $PORT \
      --enforce-eager \
      --disable-log-requests \
      --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
      --tensor-parallel-size $PREFILLER_TP_SIZE \
      --kv-transfer-config '{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_producer\",\"kv_connector_extra_config\": {\"discard_partial_chunks\": false, \"lmcache_rpc_port\": \"producer$((i+1))\"}}'
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
    # Calculate GPU ID - distribute across available GPUs, starting from after prefill instances (like run_accuracy_test.sh)
    if [ "$DECODER_TP_SIZE" -eq 1 ]; then
      GPU_ID=$(((i + NUM_PREFILL_INSTANCES) % $(get_num_gpus)))
      GPU_DEVICES="$GPU_ID"
    else
      # For TP > 1, use consecutive GPUs starting from calculated base
      base_gpu=$(((i + NUM_PREFILL_INSTANCES) % $(get_num_gpus)))
      GPU_DEVICES=$(seq -s, $base_gpu $((base_gpu + DECODER_TP_SIZE - 1)))
    fi
    
    # Calculate port number (base port + instance number)
    PORT=$((8200 + i))

    echo "Starting decode instance $i on GPU(s) $GPU_DEVICES, port $PORT (TP size: $DECODER_TP_SIZE)"

    # Build the command with or without model-specific args
    BASE_CMD="UCX_TLS=cuda_ipc,cuda_copy,tcp \
      LMCACHE_USE_EXPERIMENTAL=True \
      VLLM_ENABLE_V1_MULTIPROCESSING=1 \
      VLLM_WORKER_MULTIPROC_METHOD=spawn \
      CUDA_VISIBLE_DEVICES=$GPU_DEVICES \
      vllm serve $model_name \
      --port $PORT \
      --enforce-eager \
      --disable-log-requests \
      --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
      --tensor-parallel-size $DECODER_TP_SIZE \
      --kv-transfer-config '{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_consumer\",\"kv_connector_extra_config\": {\"discard_partial_chunks\": false, \"lmcache_rpc_port\": \"consumer$((i+1))\"}}'
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

  # Create timestamped results directory
  TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
  RESULTS_DIR="${GIT_ROOT}/tests/v1/kv_connector/nixl_integration/results_lmcache_${TIMESTAMP}"
  mkdir -p "$RESULTS_DIR"

  echo "Results will be saved to: $RESULTS_DIR"

  # Record metadata
  cat > "$RESULTS_DIR/metadata.json" <<EOF
{
  "model_name": "$model_name",
  "dataset_name": "$DATASET_NAME",
  "num_prefill_instances": $NUM_PREFILL_INSTANCES,
  "num_decode_instances": $NUM_DECODE_INSTANCES,
  "prefiller_tp_size": $PREFILLER_TP_SIZE,
  "decoder_tp_size": $DECODER_TP_SIZE,
  "gpu_memory_utilization": $GPU_MEMORY_UTILIZATION,
  "num_prompts": $NUM_PROMPT,
  "request_rates": [$(IFS=','; echo "${REQUEST_RATES[*]}")],
  "mode": "lmcache",
  "timestamp": "$TIMESTAMP"
}
EOF

  # Run benchmarks for each request rate
  for rate in "${REQUEST_RATES[@]}"; do
    echo "Running benchmark with request rate: $rate req/s"
    
    # Run the benchmark
    OUTPUT_FILE="$RESULTS_DIR/vllm-${rate}qps-${DATASET_NAME}-result.json"
    python3 ${GIT_ROOT}/benchmarks/benchmark_serving.py \
      --backend vllm \
      --host localhost \
      --port ${PROXY_PORT} \
      --model "$model_name" \
      --dataset-name "$DATASET_NAME" \
      --num-prompts $NUM_PROMPT \
      --request-rate "$rate" \
      --save-detailed "$OUTPUT_FILE" \
      --metadata "$RESULTS_DIR/metadata.json"

    echo "Benchmark completed for rate $rate req/s, results saved to $OUTPUT_FILE"
    sleep 2
  done

  # Clean up before running next model
  cleanup_instances
  sleep 3
}

# Main execution
echo "Starting LMCache benchmark tests..."

# Check if LMCache configs exist
if [ ! -d "$LMCACHE_DIR" ]; then
  echo "ERROR: LMCache directory not found: $LMCACHE_DIR"
  exit 1
fi

if [ ! -f "$LMCACHE_DIR/configs/lmcache-prefiller-config.yaml" ]; then
  echo "ERROR: LMCache prefiller config not found: $LMCACHE_DIR/configs/lmcache-prefiller-config.yaml"
  exit 1
fi

if [ ! -f "$LMCACHE_DIR/configs/lmcache-decoder-config.yaml" ]; then
  echo "ERROR: LMCache decoder config not found: $LMCACHE_DIR/configs/lmcache-decoder-config.yaml"
  exit 1
fi

# Check dependencies
echo "Checking LMCache dependencies..."
python3 -c "import lmcache" || { echo "LMCache not installed. Please install it first."; exit 1; }

# Run tests for each model
for model in "${MODELS[@]}"; do
  run_tests_for_model "$model"
done

echo "All LMCache benchmark tests completed!"
