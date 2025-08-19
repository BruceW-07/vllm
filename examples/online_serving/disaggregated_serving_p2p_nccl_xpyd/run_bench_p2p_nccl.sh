#!/bin/bash
set -xe

# Dataset name
DATASET_NAME=${1:-"sharegpt"}

# Models to run
MODELS=(
    "/workspace/models/Qwen3-8B"
)

# Prefill/Decode server数量与TP设置
NUM_PREFILL_INSTANCES=${NUM_PREFILL_INSTANCES:-1}
NUM_DECODE_INSTANCES=${NUM_DECODE_INSTANCES:-1}
PREFILLER_TP_SIZE=${PREFILLER_TP_SIZE:-1}
DECODER_TP_SIZE=${DECODER_TP_SIZE:-1}

# GPU分配与内存设置
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.8}
START_GPU_ID=${START_GPU_ID:-0}

# 其他配置
NUM_PROMPT="100"
REQUEST_RATES=(3.0 2.5 2.0 1.5 1.0 0.5)
CUSTOM_OUTPUT_LEN=${CUSTOM_OUTPUT_LEN:-128}

trap 'kill $(jobs -pr)' SIGINT SIGTERM EXIT

wait_for_server() {
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

cleanup_instances() {
  echo "Cleaning up any running vLLM instances..."
  pkill -f "vllm serve" || true
  sleep 2
}

get_model_args() {
  local model_name=$1
  local extra_args=""
  if [[ "$model_name" == "deepseek-ai/deepseek-vl2-tiny" ]]; then
    extra_args="--hf_overrides '{\"architectures\": [\"DeepseekVLV2ForCausalLM\"]}' --trust-remote-code"
  fi
  echo "$extra_args"
}

get_num_gpus() {
  nvidia-smi --query-gpu=name --format=csv,noheader | wc -l
}

run_tests_for_model() {
  local model_name="$1"
  echo "================================"
  echo "Testing model: $model_name (P2P NCCL Mode)"
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

  local model_args=$(get_model_args "$model_name")

  PREFILL_PORTS=()
  DECODE_PORTS=()
  PREFILL_HOSTS=()
  DECODE_HOSTS=()

  # 启动Prefill实例
  for i in $(seq 0 $((NUM_PREFILL_INSTANCES-1))); do
    GPU_ID=$(((START_GPU_ID + i) % $(get_num_gpus)))
    GPU_DEVICES="$GPU_ID"
    PORT=$((9157 + i))
    KV_PORT=$((21001 + i))
    echo "Starting prefill instance $i on GPU $GPU_DEVICES, port $PORT, KV port $KV_PORT"
    BASE_CMD="CUDA_VISIBLE_DEVICES=$GPU_DEVICES VLLM_USE_V1=1 vllm serve $model_name \
      --enforce-eager \
      --host 0.0.0.0 \
      --port $PORT \
      --tensor-parallel-size $PREFILLER_TP_SIZE \
      --seed 1024 \
      --dtype float16 \
      --max-model-len 10000 \
      --max-num-batched-tokens 10000 \
      --max-num-seqs 256 \
      --trust-remote-code \
      --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
      --disable-log-requests \
      --no-enable-prefix-caching \
      --no-enable-chunked-prefill \
      --kv-transfer-config '{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_producer\",\"kv_buffer_size\":\"1e1\",\"kv_port\":\"$KV_PORT\",\"kv_connector_extra_config\":{\"proxy_ip\":\"0.0.0.0\",\"proxy_port\":\"30777\",\"http_port\":\"$PORT\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"16\"}}'"
    if [ -n "$model_args" ]; then
      FULL_CMD="$BASE_CMD $model_args"
    else
      FULL_CMD="$BASE_CMD"
    fi
    eval "$FULL_CMD &"
    PREFILL_HOSTS+=("localhost")
    PREFILL_PORTS+=($PORT)
  done

  # 启动Decode实例
  for i in $(seq 0 $((NUM_DECODE_INSTANCES-1))); do
    GPU_ID=$(((START_GPU_ID + i + NUM_PREFILL_INSTANCES) % $(get_num_gpus)))
    GPU_DEVICES="$GPU_ID"
    PORT=$((9257 + i))
    KV_PORT=$((22001 + i))
    echo "Starting decode instance $i on GPU $GPU_DEVICES, port $PORT, KV port $KV_PORT"
    BASE_CMD="CUDA_VISIBLE_DEVICES=$GPU_DEVICES VLLM_USE_V1=1 vllm serve $model_name \
      --enforce-eager \
      --host 0.0.0.0 \
      --port $PORT \
      --tensor-parallel-size $DECODER_TP_SIZE \
      --seed 1024 \
      --dtype float16 \
      --max-model-len 10000 \
      --max-num-batched-tokens 10000 \
      --max-num-seqs 256 \
      --trust-remote-code \
      --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
      --disable-log-requests \
      --no-enable-prefix-caching \
      --no-enable-chunked-prefill \
      --kv-transfer-config '{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_consumer\",\"kv_buffer_size\":\"8e9\",\"kv_port\":\"$KV_PORT\",\"kv_connector_extra_config\":{\"proxy_ip\":\"0.0.0.0\",\"proxy_port\":\"30777\",\"http_port\":\"$PORT\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"16\"}}'"
    if [ -n "$model_args" ]; then
      FULL_CMD="$BASE_CMD $model_args"
    else
      FULL_CMD="$BASE_CMD"
    fi
    eval "$FULL_CMD &"
    DECODE_HOSTS+=("localhost")
    DECODE_PORTS+=($PORT)
  done

  # 等待所有服务启动
  for PORT in "${PREFILL_PORTS[@]}"; do
    echo "Waiting for prefill instance on port $PORT to start..."
    wait_for_server $PORT
  done
  for PORT in "${DECODE_PORTS[@]}"; do
    echo "Waiting for decode instance on port $PORT to start..."
    wait_for_server $PORT
  done

  # 启动proxy server
  PROXY_PORT=30777
  PROXY_CMD="python disagg_proxy_p2p_nccl_xpyd.py --port $PROXY_PORT"
  echo "Starting proxy server with command: $PROXY_CMD"
  $PROXY_CMD &
  sleep 5

  SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  RESULTS_BASE_DIR="$SCRIPT_DIR/results/$DATASET_NAME"
  RESULTS_DIR="$RESULTS_BASE_DIR/$TIMESTAMP"
  mkdir -p "$RESULTS_DIR"
  echo "Results will be saved to: $RESULTS_DIR"

  for REQUEST_RATE in "${REQUEST_RATES[@]}"; do
    TS=$(date +%Y%m%d_%H%M%S)
    echo "Running vllm bench serve for $model_name with request rate $REQUEST_RATE"
    CONFIG_SUFFIX="pf${NUM_PREFILL_INSTANCES}d${NUM_DECODE_INSTANCES}tp${PREFILLER_TP_SIZE}x${DECODER_TP_SIZE}"
    MODEL_NAME=$(basename "$model_name")
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
      --custom-output-len $CUSTOM_OUTPUT_LEN \
      --metadata \
        "num_prefill_instances=$NUM_PREFILL_INSTANCES" \
        "num_decode_instances=$NUM_DECODE_INSTANCES" \
        "prefiller_tp_size=$PREFILLER_TP_SIZE" \
        "decoder_tp_size=$DECODER_TP_SIZE" \
        "dataset_name=$DATASET_NAME" \
        "model_name=$MODEL_NAME" \
        "config_suffix=$CONFIG_SUFFIX" \
        "gpu_memory_utilization=$GPU_MEMORY_UTILIZATION" \
        "kv_connector=P2pNcclConnector" \
        "prefix_caching=disabled" \
        "chunked_prefill=disabled"
    echo "Completed benchmark with request rate $REQUEST_RATE"
    sleep 10
  done

  cleanup_instances
  sleep 3
}

for idx in "${!MODELS[@]}"; do
  model_name="${MODELS[$idx]}"
  run_tests_for_model "$model_name"
done

echo "All tests completed!"
