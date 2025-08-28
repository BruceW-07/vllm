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

# Define paths relative to script location
simple_results_path="./simple/results"
p2p_nccl_results_path="./p2p_nccl/results"

# Check if result paths exist
if [ ! -d "${simple_results_path}" ]; then
    echo "Warning: Simple results path ${simple_results_path} does not exist."
fi

if [ ! -d "${p2p_nccl_results_path}" ]; then
    echo "Warning: P2P NCCL results path ${p2p_nccl_results_path} does not exist."
fi

# 1. Run plot_latency_breakdown.py for simple configuration
echo ""
echo "1. Generating latency breakdown plot for simple configuration..."
run_plot_script "plot_latency_breakdown.py" "${simple_results_path}"

# 2. Run plot_latency_breakdown.py for p2p_nccl configuration
echo ""
echo "2. Generating latency breakdown plot for p2p_nccl configuration..."
run_plot_script "plot_latency_breakdown.py" "${p2p_nccl_results_path}"

# 3. Run plot_latency_rps_per_gpu_comparison.py
echo ""
echo "3. Generating latency RPS per GPU comparison plots..."
run_plot_script "plot_latency_rps_per_gpu_comparison.py" "${simple_results_path}" "${p2p_nccl_results_path}"

# 4. Run plot_slo_attainment_rps_per_gpu_comparison.py
echo ""
echo "4. Generating SLO attainment RPS per GPU comparison plots..."
run_plot_script "plot_slo_attainment_rps_per_gpu_comparison.py" "${simple_results_path}" "${p2p_nccl_results_path}"

echo ""
echo "All plots have been generated successfully!"