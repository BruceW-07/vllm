#!/bin/bash
set -e

# Simple script to run vLLM benchmark and generate plots
# Usage: ./run_benchmark_and_plot.sh [dataset_name]

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Get dataset name from command line argument or use default
DATASET_NAME=${1:-"sharegpt"}

echo "=========================================="
echo "Running vLLM Benchmark and Plotting"
echo "Dataset: $DATASET_NAME"
echo "=========================================="

# Step 1: Run benchmark
echo "Step 1: Running benchmark..."
bash "$SCRIPT_DIR/run_bench.sh" "$DATASET_NAME"

# Step 2: Generate plots
echo "Step 2: Generating plots..."
RESULTS_DIR="$SCRIPT_DIR/results/$DATASET_NAME"

if [[ ! -d "$RESULTS_DIR" ]]; then
    echo "Error: Results directory not found: $RESULTS_DIR"
    exit 1
fi

python3 "$SCRIPT_DIR/benchmark_plotter.py" plot --dir "$RESULTS_DIR"

echo "=========================================="
echo "Pipeline completed!"
echo "Results: $SCRIPT_DIR/results/"
echo "Plots: $SCRIPT_DIR/plots/"
echo "=========================================="
