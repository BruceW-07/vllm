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

# Plotting parameters
TTFT_SLO=${TTFT_SLO:-125}  # milliseconds
TPOT_SLO=${TPOT_SLO:-200}  # milliseconds
TARGET_ATTAINMENT=${TARGET_ATTAINMENT:-90}  # percentage

echo "=========================================="
echo "Running vLLM Benchmark and Plotting"
echo "Dataset: $DATASET_NAME"
echo "Prefill instances: $NUM_PREFILL_INSTANCES"
echo "Decode instances: $NUM_DECODE_INSTANCES"
echo "Prefiller TP size: $PREFILLER_TP_SIZE"
echo "Decoder TP size: $DECODER_TP_SIZE"
echo "TTFT SLO: ${TTFT_SLO}ms"
echo "TPOT SLO: ${TPOT_SLO}ms"
echo "Target attainment: ${TARGET_ATTAINMENT}%"
echo "=========================================="

# Step 1: Run benchmark
echo "Step 1: Running benchmark..."

# Pass environment variables to the benchmark script
NUM_PREFILL_INSTANCES="$NUM_PREFILL_INSTANCES" \
NUM_DECODE_INSTANCES="$NUM_DECODE_INSTANCES" \
PREFILLER_TP_SIZE="$PREFILLER_TP_SIZE" \
DECODER_TP_SIZE="$DECODER_TP_SIZE" \
bash "$SCRIPT_DIR/run_bench_nixl.sh" "$DATASET_NAME"

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

# Create plots directory in the same timestamped folder
PLOTS_DIR="$LATEST_RESULTS_DIR/plots"
mkdir -p "$PLOTS_DIR"

# Set environment variable to tell the plotter where to save plots
PLOT_OUTPUT_DIR="$PLOTS_DIR" python3 "$SCRIPT_DIR/benchmark_plotter.py" \
    --dir "$LATEST_RESULTS_DIR" \
    --ttft-slo "$TTFT_SLO" \
    --tpot-slo "$TPOT_SLO" \
    --target "$TARGET_ATTAINMENT"

echo "=========================================="
echo "Pipeline completed!"
echo "Results: $LATEST_RESULTS_DIR"
echo "Plots: $PLOTS_DIR"
echo "=========================================="
