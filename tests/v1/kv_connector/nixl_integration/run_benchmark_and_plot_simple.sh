#!/bin/bash
set -e

# Simple script to run vLLM benchmark (without PD separation) and generate plots
# Usage: ./run_benchmark_and_plot_simple.sh [dataset_name]

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Get dataset name from command line argument or use default
DATASET_NAME=${1:-"sharegpt"}

# Simple configuration (no PD separation)
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-2}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.8}

# Plotting parameters
TTFT_SLO=${TTFT_SLO:-125}  # milliseconds
TPOT_SLO=${TPOT_SLO:-200}  # milliseconds
TARGET_ATTAINMENT=${TARGET_ATTAINMENT:-90}  # percentage

echo "=========================================="
echo "Running vLLM Benchmark and Plotting (Simple Mode)"
echo "Dataset: $DATASET_NAME"
echo "Tensor Parallel size: $TENSOR_PARALLEL_SIZE"
echo "GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
echo "TTFT SLO: ${TTFT_SLO}ms"
echo "TPOT SLO: ${TPOT_SLO}ms"
echo "Target attainment: ${TARGET_ATTAINMENT}%"
echo "=========================================="

# Step 1: Run benchmark
echo "Step 1: Running benchmark..."

# Pass environment variables to the benchmark script
TENSOR_PARALLEL_SIZE="$TENSOR_PARALLEL_SIZE" \
GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
bash "$SCRIPT_DIR/run_bench_simple.sh" "$DATASET_NAME"

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
