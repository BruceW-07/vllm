#!/bin/bash

# Script to generate all plots by calling the individual plotting scripts.
#
# This script will call the following plotting scripts:
# 1. plot_latency_breakdown.py
# 2. plot_latency_rps_per_gpu_comparison.py
# 3. plot_slo_attainment_rps_per_gpu_comparison.py
#
# The input data paths are:
# - Simple configuration results: ./simple/results
# - P2P NCCL configuration results: ./p2p_nccl/results

# Function to run a plotting script with the given arguments
run_plot_script() {
    local script_name=$1
    shift
    local script_path="../ploters/${script_name}"
    
    echo "Running: python3 ${script_path} $@"
    python3 "${script_path}" "$@"
    
    if [ $? -ne 0 ]; then
        echo "Error running ${script_name}"
        return 1
    fi
    return 0
}

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the directory where this script is located
cd "${SCRIPT_DIR}" || exit 1

echo "Generating all plots..."

# Define datasets to process
datasets=("random-512-64.sh" "sharegpt.sh" "gsm8k.sh" "human_eval.sh")

# Process each dataset
for dataset in "${datasets[@]}"; do
    echo ""
    echo "Processing dataset: ${dataset}"
    echo "========================"
    
    # Remove .sh extension for directory names
    dataset_name="${dataset%.sh}"
    
    # Define paths relative to script location for this dataset
    simple_results_path="./simple/results/${dataset_name}"
    p2p_nccl_results_path="./p2p_nccl/results/${dataset_name}"
    
    # Define output directories for this dataset
    output_dir="../plots/${dataset_name}"
    
    # Create output directory if it doesn't exist
    mkdir -p "${output_dir}"
    
    # Check if result paths exist
    if [ ! -d "${simple_results_path}" ]; then
        echo "Warning: Simple results path ${simple_results_path} does not exist."
    fi
    
    if [ ! -d "${p2p_nccl_results_path}" ]; then
        echo "Warning: P2P NCCL results path ${p2p_nccl_results_path} does not exist."
    fi
    
    # 1. Run plot_latency_breakdown.py for p2p_nccl configuration
    echo ""
    echo "1. Generating latency breakdown plot for p2p_nccl configuration..."
    run_plot_script "plot_latency_breakdown.py" "${p2p_nccl_results_path}" --num-gpus 2 --plot-type latency --output "${output_dir}/p2p_nccl_latency_breakdown.png"
    run_plot_script "plot_latency_breakdown.py" "${p2p_nccl_results_path}" --num-gpus 2 --plot-type ttft --output "${output_dir}/p2p_nccl_ttft_breakdown.png"
    
    # 2. Run plot_latency_rps_per_gpu_comparison.py
    echo ""
    echo "2. Generating latency RPS per GPU comparison plots..."
    run_plot_script "plot_latency_rps_per_gpu_comparison.py" "${simple_results_path}" "${p2p_nccl_results_path}" --output "${output_dir}/latency_rps_per_gpu_comparison.png"
    
    # 3. Run plot_slo_attainment_rps_per_gpu_comparison.py
    echo ""
    echo "3. Generating SLO attainment RPS per GPU comparison plots..."
    run_plot_script "plot_slo_attainment_rps_per_gpu_comparison.py" "${simple_results_path}" "${p2p_nccl_results_path}" --ttft-limit 100 --tpot-limit 17 --output "${output_dir}/slo_attainment_rps_per_gpu_comparison.png"
    
    echo ""
    echo "Finished processing dataset: ${dataset}"
done

echo ""
echo "All plots have been generated successfully for all datasets!"