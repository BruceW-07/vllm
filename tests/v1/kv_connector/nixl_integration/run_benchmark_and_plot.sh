#!/bin/bash
set -e

# Simple script to run vLLM benchmark and generate plots
# Usage: ./run_benchmark_and_plot.sh [dataset_name]

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Get dataset name from command line argument or use default
DATASET_NAME=${1:-"sharegpt"}

# Number of prefill and decode instances to create
NUM_PREFILL_INSTANCES=${NUM_PREFILL_INSTANCES:-1} # Default to 1
NUM_DECODE_INSTANCES=${NUM_DECODE_INSTANCES:-1}   # Default to 1
PREFILLER_TP_SIZE=${PREFILLER_TP_SIZE:-1}
DECODER_TP_SIZE=${DECODER_TP_SIZE:-1}

echo "=========================================="
echo "Running vLLM Benchmark and Plotting"
echo "Dataset: $DATASET_NAME"
echo "Prefill instances: $NUM_PREFILL_INSTANCES"
echo "Decode instances: $NUM_DECODE_INSTANCES"
echo "Prefiller TP size: $PREFILLER_TP_SIZE"
echo "Decoder TP size: $DECODER_TP_SIZE"
echo "=========================================="

# Step 1: Run benchmark
echo "Step 1: Running benchmark..."

# Pass environment variables to the benchmark script
NUM_PREFILL_INSTANCES="$NUM_PREFILL_INSTANCES" \
NUM_DECODE_INSTANCES="$NUM_DECODE_INSTANCES" \
PREFILLER_TP_SIZE="$PREFILLER_TP_SIZE" \
DECODER_TP_SIZE="$DECODER_TP_SIZE" \
bash "$SCRIPT_DIR/run_bench.sh" "$DATASET_NAME"

# Step 2: Generate plots
echo "Step 2: Generating plots..."
RESULTS_BASE_DIR="$SCRIPT_DIR/results/$DATASET_NAME"

if [[ ! -d "$RESULTS_BASE_DIR" ]]; then
    echo "Error: Results base directory not found: $RESULTS_BASE_DIR"
    exit 1
fi

# Find the most recent timestamped directory
LATEST_RESULTS_DIR=$(find "$RESULTS_BASE_DIR" -maxdepth 1 -type d -name "20*" | sort | tail -1)

if [[ -z "$LATEST_RESULTS_DIR" ]]; then
    echo "Error: No timestamped results directory found in $RESULTS_BASE_DIR"
    exit 1
fi

echo "Using results from: $LATEST_RESULTS_DIR"

python3 "$SCRIPT_DIR/benchmark_plotter.py" plot --dir "$LATEST_RESULTS_DIR"

echo "=========================================="
echo "Pipeline completed!"
echo "Results: $SCRIPT_DIR/results/"
echo "Plots: $SCRIPT_DIR/plots/"
echo "=========================================="
