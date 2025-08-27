#!/usr/bin/env python3
"""
Script to read benchmark results from JSON files and plot latency comparison
between simple and p2p_nccl configurations for TTFT and TPOT metrics when 
request rate per GPU is the same.

Usage:
    python plot_latency_rps_per_gpu_comparison.py <simple_folder> <p2p_nccl_folder> [--output OUTPUT_FILE]
    
    Example:
        python plot_latency_rps_per_gpu_comparison.py ./simple ./p2p_nccl
        python plot_latency_rps_per_gpu_comparison.py ./simple ./p2p_nccl --output comparison.png
        
    Arguments:
        simple_folder     Path to the simple configuration results folder
        p2p_nccl_folder   Path to the p2p_nccl configuration results folder
        --output, -o      Output file path for the plot (PNG format)
                          Default: latency_rps_per_gpu_comparison.png
"""

import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np


def extract_request_rate_from_filename(filename):
    """Extract request rate from filename."""
    # Assuming filename format like: label-request_rateqps-model-date.json
    parts = filename.split('-')
    for part in parts:
        if 'qps' in part:
            # Extract the numeric part before 'qps'
            rate_str = part.replace('qps', '')
            try:
                return float(rate_str)
            except ValueError:
                continue
    return None


def read_benchmark_data(folder_path):
    """
    Read all JSON files in a folder and extract metrics data.
    
    Args:
        folder_path (str): Path to the folder containing JSON benchmark results
    
    Returns:
        dict: A dictionary with request_rate as keys and a dict of metrics as values.
              The metrics dict has percentiles ('p90', 'p95', 'p99') as keys, and tuples
              of (ttft_ms, tpot_ms) as values.
    """
    data = {}
    
    if not os.path.exists(folder_path):
        print(f"Warning: Folder {folder_path} does not exist.")
        return data
        
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            
            try:
                with open(file_path, 'r') as f:
                    json_data = json.load(f)
                
                # Try to get request_rate from JSON data first
                request_rate = None
                if 'request_rate' in json_data:
                    request_rate = json_data['request_rate']
                    # Convert "inf" to a large number for sorting purposes
                    if request_rate == "inf":
                        request_rate = float('inf')
                    else:
                        request_rate = float(request_rate)
                
                # If not in JSON data, try to extract from filename
                if request_rate is None:
                    request_rate = extract_request_rate_from_filename(filename)
                
                if request_rate is None:
                    print(f"Warning: Could not determine request rate for {filename}")
                    continue
                
                # Extract metrics for different percentiles
                metrics = {}
                
                # Process each percentile
                percentiles = ['p90', 'p95', 'p99']
                for percentile in percentiles:
                    # Extract metrics for this percentile
                    ttft_ms = json_data.get(f'{percentile}_ttft_ms')
                    tpot_ms = json_data.get(f'{percentile}_tpot_ms')
                    
                    if ttft_ms is not None and tpot_ms is not None:
                        metrics[percentile] = (float(ttft_ms), float(tpot_ms))
                
                # If we have at least one percentile, store the data
                if metrics:
                    data[request_rate] = metrics
                else:
                    print(f"Warning: Could not find valid metrics in {filename}")
                
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue
    
    return data


def plot_performance_comparison(simple_data, p2p_nccl_data, output_file=None):
    """
    Plot latency comparison between simple and p2p_nccl configurations for TTFT and TPOT.
    
    Args:
        simple_data (dict): Data from simple configuration
        p2p_nccl_data (dict): Data from p2p_nccl configuration
        output_file (str): Output file path for the plot
    """
    # Plot for each percentile
    percentiles = ['p90', 'p95', 'p99']
    for percentile in percentiles:
        plot_performance_comparison_percentile(simple_data, p2p_nccl_data, percentile, output_file)


def plot_performance_comparison_percentile(simple_data, p2p_nccl_data, percentile, output_file=None):
    """
    Plot latency comparison between simple and p2p_nccl configurations for a specific percentile.
    
    Args:
        simple_data (dict): Data from simple configuration
        p2p_nccl_data (dict): Data from p2p_nccl configuration
        percentile (str): Percentile to plot (e.g., 'p90', 'p95', 'p99')
        output_file (str): Output file path for the plot
    """
    # Get all request rates from both datasets
    all_rates = set(list(simple_data.keys()) + list(p2p_nccl_data.keys()))
    # Filter out infinite rates and sort
    finite_rates = [rate for rate in all_rates if rate != float('inf')]
    sorted_rates = sorted(finite_rates)
    
    if not sorted_rates:
        print("No valid request rates found.")
        return
    
    # Extract data for plotting
    # For simple configuration
    simple_ttft = []
    simple_tpot = []
    # For p2p_nccl configuration (will be adjusted for 1p1d)
    p2p_nccl_ttft = []
    p2p_nccl_tpot = []
    # Request rates that have data in both configurations
    valid_rates = []
    
    # Adjusted rates for p2p_nccl (divided by 2 for 1p1d configuration)
    p2p_nccl_adjusted_rates = []
    
    for rate in sorted_rates:
        # Check if rate exists in both datasets and has the required percentile data
        if (rate in simple_data and rate in p2p_nccl_data and
            percentile in simple_data[rate] and percentile in p2p_nccl_data[rate]):
            valid_rates.append(rate)
            # For p2p_nccl, adjust rate by dividing by 2 (1p1d configuration)
            p2p_nccl_adjusted_rates.append(rate / 2)
            # Extract data for the specific percentile
            simple_ttft.append(simple_data[rate][percentile][0])
            simple_tpot.append(simple_data[rate][percentile][1])
            p2p_nccl_ttft.append(p2p_nccl_data[rate][percentile][0])
            p2p_nccl_tpot.append(p2p_nccl_data[rate][percentile][1])
    
    if not valid_rates:
        print(f"No common request rates found between the two configurations for {percentile}.")
        return
    
    # Convert ms to seconds
    simple_ttft_s = [ttft / 1000 for ttft in simple_ttft]
    simple_tpot_s = [tpot / 1000 for tpot in simple_tpot]
    p2p_nccl_ttft_s = [ttft / 1000 for ttft in p2p_nccl_ttft]
    p2p_nccl_tpot_s = [tpot / 1000 for tpot in p2p_nccl_tpot]
    
    # Set up the plot with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Improved colors
    simple_color = '#1f77b4'  # Blue
    p2p_nccl_color = '#ff7f0e'  # Orange
    marker_style = 'o'  # Circular markers
    
    # Plot TTFT comparison (top subplot)
    ax1.plot(valid_rates, simple_ttft_s,
             color=simple_color, marker=marker_style, linewidth=2, markersize=6,
             label='Simple')
    ax1.plot(p2p_nccl_adjusted_rates, p2p_nccl_ttft_s,
             color=p2p_nccl_color, marker=marker_style, linewidth=2, markersize=6,
             label='P2P NCCL (1p1d)')
    ax1.set_xlabel('Request Rate / GPU (reqs/s/GPU)')
    ax1.set_ylabel(f'{percentile.upper()} TTFT (s)')
    ax1.set_title(f'{percentile.upper()} TTFT Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot TPOT comparison (bottom subplot)
    ax2.plot(valid_rates, simple_tpot_s,
             color=simple_color, marker=marker_style, linewidth=2, markersize=6,
             label='Simple')
    ax2.plot(p2p_nccl_adjusted_rates, p2p_nccl_tpot_s,
             color=p2p_nccl_color, marker=marker_style, linewidth=2, markersize=6,
             label='P2P NCCL (1p1d)')
    ax2.set_xlabel('Request Rate / GPU (reqs/s/GPU)')
    ax2.set_ylabel(f'{percentile.upper()} TPOT (s)')
    ax2.set_title(f'{percentile.upper()} TPOT Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Determine output file name
    if output_file:
        # Replace the extension with the percentile
        base_name, ext = os.path.splitext(output_file)
        percentile_output_file = f"{base_name}_{percentile}{ext}"
    else:
        # Create default plots directory if it doesn't exist
        default_plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plots")
        os.makedirs(default_plots_dir, exist_ok=True)
        percentile_output_file = os.path.join(default_plots_dir, f"latency_rps_per_gpu_comparison_{percentile}.png")
    
    plt.savefig(percentile_output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {percentile_output_file}")
    plt.close()  # Close the figure to free memory


def main():
    parser = argparse.ArgumentParser(description='Plot latency results comparison with RPS/GPU')
    parser.add_argument('simple_folder', help='Path to the simple configuration results folder')
    parser.add_argument('p2p_nccl_folder', help='Path to the p2p_nccl configuration results folder')
    parser.add_argument('--output', '-o',
                        help='Output file path for the plots (PNG format). Default: ../plots/latency_rps_per_gpu_comparison_*.png')
    
    args = parser.parse_args()
    
    # Read data from both folders
    print("Reading data from simple configuration folder...")
    simple_data = read_benchmark_data(args.simple_folder)
    print(f"Found {len(simple_data)} data points in simple configuration")
    
    print("Reading data from p2p_nccl configuration folder...")
    p2p_nccl_data = read_benchmark_data(args.p2p_nccl_folder)
    print(f"Found {len(p2p_nccl_data)} data points in p2p_nccl configuration")
    
    if not simple_data and not p2p_nccl_data:
        print("No data found in either folder. Exiting.")
        return
    
    # Plot the comparison for all percentiles
    print("Generating latency comparison plots for all percentiles...")
    plot_performance_comparison(simple_data, p2p_nccl_data, args.output)


if __name__ == '__main__':
    main()