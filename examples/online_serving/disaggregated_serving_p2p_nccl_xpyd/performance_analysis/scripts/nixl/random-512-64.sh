#!/bin/bash

# =============================================================================
# Benchmark Script for vLLM Disaggregated Serving with NIXL
# =============================================================================

set -xe  # Exit on any error

# Configuration - can be overridden via environment variables
DATASET_NAME=${DATASET_NAME:-random}
RANDOM_INPUT_LEN=${RANDOM_INPUT_LEN:-512}
RANDOM_OUTPUT_LEN=${RANDOM_OUTPUT_LEN:-64}
REQUEST_RATES=${REQUEST_RATES:-"1 2 3 4 5 6 7 8 9 10 11"}
SERVER_PORT=${SERVER_PORT:-10001}

# Default configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
MODEL_PATH="/workspace/models/Llama-3.1-8B-Instruct"

# Function to run benchmarks
run_benchmarks() {
    echo "Running benchmarks..."
    
    # Create results directory with dataset and input/output length information
    local result_subdir="${DATASET_NAME}_${RANDOM_INPUT_LEN}_${RANDOM_OUTPUT_LEN}"
    mkdir -p "$RESULTS_DIR/$result_subdir"
    
    # Convert REQUEST_RATES string to array
    local request_rates_array=($REQUEST_RATES)
    
    for REQUEST_RATE in "${request_rates_array[@]}"; do
        # Use bc for floating point arithmetic and convert to integer
        # bc: command-line calculator that supports floating point operations
        # cut -d. -f1: extracts the integer part before the decimal point
        NUM_PROMPTS=$(echo "$REQUEST_RATE * 300" | bc | cut -d. -f1)
        echo "Running benchmark with request rate: $REQUEST_RATE, total prompts: $NUM_PROMPTS"
        
        # Run the benchmark
        cd "$SCRIPT_DIR" && vllm bench serve \
            --backend vllm \
            --model "$MODEL_PATH" \
            --endpoint /v1/completions \
            --dataset-name "$DATASET_NAME" \
            --random-input-len "$RANDOM_INPUT_LEN" \
            --random-output-len "$RANDOM_OUTPUT_LEN" \
            --ignore-eos \
            --metric-percentiles "90,95,99" \
            --seed 1024 \
            --trust-remote-code \
            --request-rate $REQUEST_RATE \
            --num_prompt $NUM_PROMPTS \
            --save-result \
            --save-detailed \
            --result-dir "$RESULTS_DIR/$result_subdir" \
            --port $SERVER_PORT
            
        echo "Benchmark with request rate $REQUEST_RATE and $NUM_PROMPTS prompts completed."
        sleep 10
    done
    
    echo "All benchmarks completed."
}

# Main execution
main() {
    echo "Starting benchmark script for NIXL configuration..."
    echo "Dataset: $DATASET_NAME"
    echo "Input length: $RANDOM_INPUT_LEN"
    echo "Output length: $RANDOM_OUTPUT_LEN"
    
    # Run benchmarks
    run_benchmarks
}

# Run main function
main "$@"