#!/usr/bin/env python3
"""
Script to generate all plots by calling the individual plotting scripts.

This script will call the following plotting scripts:
1. plot_latency_breakdown.py
2. plot_latency_rps_per_gpu_comparison.py
3. plot_slo_attainment_rps_per_gpu_comparison.py

The input data paths are:
- Simple configuration results: ./simple/results
- P2P NCCL configuration results: ./p2p_nccl/results
"""

import os
import subprocess
import sys

def run_plot_script(script_name, *args):
    """Run a plotting script with the given arguments."""
    script_path = os.path.join("..", "ploters", script_name)
    cmd = [sys.executable, script_path] + list(args)
    
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False
    return True

def main():
    # Define paths
    simple_results_path = os.path.join(".", "simple", "results")
    p2p_nccl_results_path = os.path.join(".", "p2p_nccl", "results")
    
    # Check if result paths exist
    if not os.path.exists(simple_results_path):
        print(f"Warning: Simple results path {simple_results_path} does not exist.")
    
    if not os.path.exists(p2p_nccl_results_path):
        print(f"Warning: P2P NCCL results path {p2p_nccl_results_path} does not exist.")
    
    # Change to the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("Generating all plots...")
    
    # 1. Run plot_latency_breakdown.py for simple configuration
    print("\n1. Generating latency breakdown plot for simple configuration...")
    run_plot_script("plot_latency_breakdown.py", simple_results_path)
    
    # 2. Run plot_latency_breakdown.py for p2p_nccl configuration
    print("\n2. Generating latency breakdown plot for p2p_nccl configuration...")
    run_plot_script("plot_latency_breakdown.py", p2p_nccl_results_path)
    
    # 3. Run plot_latency_rps_per_gpu_comparison.py
    print("\n3. Generating latency RPS per GPU comparison plots...")
    run_plot_script("plot_latency_rps_per_gpu_comparison.py", simple_results_path, p2p_nccl_results_path)
    
    # 4. Run plot_slo_attainment_rps_per_gpu_comparison.py
    print("\n4. Generating SLO attainment RPS per GPU comparison plots...")
    run_plot_script("plot_slo_attainment_rps_per_gpu_comparison.py", simple_results_path, p2p_nccl_results_path)
    
    print("\nAll plots have been generated successfully!")

if __name__ == "__main__":
    main()