#!/usr/bin/env python3
"""
Script to read benchmark results from JSON files and plot performance comparison
between simple and p2p_nccl configurations.

Usage:
    python plot_disagg_benchmark_comparison.py <simple_folder> <p2p_nccl_folder> [--output OUTPUT_FILE]
    
    Example:
        python plot_disagg_benchmark_comparison.py ./simple ./p2p_nccl
        python plot_disagg_benchmark_comparison.py ./simple ./p2p_nccl --output comparison.png
        
    Arguments:
        simple_folder     Path to the simple configuration results folder
        p2p_nccl_folder   Path to the p2p_nccl configuration results folder
        --output, -o      Output file path for the plot (PNG format)
                          Default: benchmark_comparison.png
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
    Read all JSON files in a folder and extract p90_ttft_ms and p90_tpot_ms data.
    
    Returns:
        dict: A dictionary with request_rate as keys and (p90_ttft_ms, p90_tpot_ms) as values
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
                
                # Extract p90 metrics
                p90_ttft_ms = json_data.get('p90_ttft_ms')
                p90_tpot_ms = json_data.get('p90_tpot_ms')
                
                # If p90 metrics don't exist, try p99 metrics
                if p90_ttft_ms is None:
                    p90_ttft_ms = json_data.get('p99_ttft_ms')
                if p90_tpot_ms is None:
                    p90_tpot_ms = json_data.get('p99_tpot_ms')
                
                # If still None, skip this file
                if p90_ttft_ms is None or p90_tpot_ms is None:
                    print(f"Warning: Could not find p90 or p99 metrics in {filename}")
                    continue
                
                data[request_rate] = (float(p90_ttft_ms), float(p90_tpot_ms))
                
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue
    
    return data


def plot_performance_comparison(simple_data, p2p_nccl_data, output_file=None):
    """
    Plot performance comparison between simple and p2p_nccl configurations.
    
    Args:
        simple_data (dict): Data from simple configuration
        p2p_nccl_data (dict): Data from p2p_nccl configuration
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
    # For p2p_nccl configuration
    p2p_nccl_ttft = []
    p2p_nccl_tpot = []
    # Request rates that have data in both configurations
    valid_rates = []
    
    for rate in sorted_rates:
        # Check if rate exists in both datasets
        if rate in simple_data and rate in p2p_nccl_data:
            valid_rates.append(rate)
            simple_ttft.append(simple_data[rate][0])
            simple_tpot.append(simple_data[rate][1])
            p2p_nccl_ttft.append(p2p_nccl_data[rate][0])
            p2p_nccl_tpot.append(p2p_nccl_data[rate][1])
    
    if not valid_rates:
        print("No common request rates found between the two configurations.")
        return
    
    # Convert ms to seconds
    simple_ttft_s = [ttft / 1000 for ttft in simple_ttft]
    simple_tpot_s = [tpot / 1000 for tpot in simple_tpot]
    p2p_nccl_ttft_s = [ttft / 1000 for ttft in p2p_nccl_ttft]
    p2p_nccl_tpot_s = [tpot / 1000 for tpot in p2p_nccl_tpot]
    
    # Set up the plot with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Colors and markers as specified
    existing_color = 'blue'      # Existing systems
    prefill_color = 'orange'     # Prefill-only
    decode_color = 'green'       # Decode-only
    marker_style = 'o'           # Circular markers
    
    # Plot TTFT comparison (top subplot)
    ax1.plot(valid_rates, simple_ttft_s, 
             color=existing_color, marker=marker_style, linewidth=2, markersize=6,
             label='Existing systems')
    ax1.plot(valid_rates, p2p_nccl_ttft_s, 
             color=prefill_color, marker=marker_style, linewidth=2, markersize=6,
             label='Prefill-only')
    ax1.set_xlabel('Rate(reqs/s)')
    ax1.set_ylabel('P90 TTFT(s)')
    ax1.set_title('P90 TTFT Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot TPOT comparison (bottom subplot)
    ax2.plot(valid_rates, simple_tpot_s, 
             color=existing_color, marker=marker_style, linewidth=2, markersize=6,
             label='Existing systems')
    ax2.plot(valid_rates, p2p_nccl_tpot_s, 
             color=decode_color, marker=marker_style, linewidth=2, markersize=6,
             label='Decode-only')
    ax2.set_xlabel('Rate(reqs/s)')
    ax2.set_ylabel('P90 TPOT(s)')
    ax2.set_title('P90 TPOT Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Set integer ticks for x-axis
    ax1.set_xticks(valid_rates)
    ax2.set_xticks(valid_rates)
    
    plt.tight_layout()
    
    # Use default output file name if none provided
    if not output_file:
        output_file = "benchmark_comparison.png"
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot benchmark results comparison')
    parser.add_argument('simple_folder', help='Path to the simple configuration results folder')
    parser.add_argument('p2p_nccl_folder', help='Path to the p2p_nccl configuration results folder')
    parser.add_argument('--output', '-o', default='benchmark_comparison.png',
                        help='Output file path for the plot (PNG format). Default: benchmark_comparison.png')
    
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
    
    # Plot the comparison
    print("Generating performance comparison plot...")
    plot_performance_comparison(simple_data, p2p_nccl_data, args.output)


if __name__ == '__main__':
    main()