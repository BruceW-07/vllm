#!/usr/bin/env python3
"""
Script to read benchmark results from JSON files and plot performance comparison
between simple and p2p_nccl configurations for p90, p95, and p99 percentiles.

Usage:
    python plot_disagg_benchmark_comparison.py <simple_folder> <p2p_nccl_folder> [--output OUTPUT_FILE]
    
    Example:
        python plot_disagg_benchmark_comparison.py ./simple ./p2p_nccl
        python plot_disagg_benchmark_comparison.py ./simple ./p2p_nccl --output comparison.png
        
    Arguments:
        simple_folder     Path to the simple configuration results folder
        p2p_nccl_folder   Path to the p2p_nccl configuration results folder
        --output, -o      Base output file path for the plots (PNG format)
                          Default: benchmark_comparison.png
                          This will generate three files: benchmark_comparison_p90.png,
                          benchmark_comparison_p95.png, and benchmark_comparison_p99.png
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
    Read all JSON files in a folder and extract p90, p95, and p99 metrics data.
    
    Args:
        folder_path (str): Path to the folder containing JSON benchmark results
    
    Returns:
        dict: A dictionary with request_rate as keys and a dict of metrics as values.
              The metrics dict has percentiles ('p90', 'p95', 'p99') as keys, and tuples
              of (ttft_ms, tpot_ms, simplified_ttft_ms) as values.
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
                    prefill_queue_time_ms = json_data.get(f'{percentile}_prefill_queue_time_ms')
                    prefill_execute_time_ms = json_data.get(f'{percentile}_prefill_execute_time_ms')
                    decode_execute_time_ms = json_data.get(f'{percentile}_decode_execute_time_ms')
                    decode_queue_time_ms = json_data.get(f'{percentile}_decode_queue_time_ms')
                    
                    if ttft_ms is not None and tpot_ms is not None:
                        # Calculate simplified TTFT for this percentile
                        if None not in [prefill_queue_time_ms, prefill_execute_time_ms, decode_execute_time_ms]:
                            simplified_ttft = (prefill_queue_time_ms +
                                              prefill_execute_time_ms +
                                              decode_execute_time_ms)
                            # Calculate p2pnccl without kvtransfer TTFT (simplified_ttft + decode_queue)
                            if decode_queue_time_ms is not None:
                                no_kvtransfer_ttft = simplified_ttft + decode_queue_time_ms
                                metrics[percentile] = (float(ttft_ms), float(tpot_ms), float(simplified_ttft), float(no_kvtransfer_ttft))
                            else:
                                metrics[percentile] = (float(ttft_ms), float(tpot_ms), float(simplified_ttft))
                        else:
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
    Plot performance comparison between simple and p2p_nccl configurations for all percentiles.
    
    Args:
        simple_data (dict): Data from simple configuration
        p2p_nccl_data (dict): Data from p2p_nccl configuration
        output_file (str): Base output file path for the plots
    """
    # Plot for each percentile
    percentiles = ['p90', 'p95', 'p99']
    for percentile in percentiles:
        plot_performance_comparison_percentile(simple_data, p2p_nccl_data, percentile, output_file)


def plot_performance_comparison_percentile(simple_data, p2p_nccl_data, percentile, output_file=None):
    """
    Plot performance comparison between simple and p2p_nccl configurations for a specific percentile.
    
    Args:
        simple_data (dict): Data from simple configuration
        p2p_nccl_data (dict): Data from p2p_nccl configuration
        percentile (str): Percentile to plot (e.g., 'p90', 'p95', 'p99')
        output_file (str): Base output file path for the plot
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
    simple_simplified_ttft = []
    # For p2p_nccl configuration
    p2p_nccl_ttft = []
    p2p_nccl_tpot = []
    p2p_nccl_simplified_ttft = []
    p2p_nccl_no_kvtransfer_ttft = []
    # Request rates that have data in both configurations
    valid_rates = []
    
    for rate in sorted_rates:
        # Check if rate exists in both datasets and has the required percentile data
        if (rate in simple_data and rate in p2p_nccl_data and
            percentile in simple_data[rate] and percentile in p2p_nccl_data[rate]):
            valid_rates.append(rate)
            # Extract data for the specific percentile
            simple_ttft.append(simple_data[rate][percentile][0])
            simple_tpot.append(simple_data[rate][percentile][1])
            p2p_nccl_ttft.append(p2p_nccl_data[rate][percentile][0])
            p2p_nccl_tpot.append(p2p_nccl_data[rate][percentile][1])
            
            # Extract simplified TTFT data if available
            if len(simple_data[rate][percentile]) > 2:
                simple_simplified_ttft.append(simple_data[rate][percentile][2])
            else:
                simple_simplified_ttft.append(None)
                
            if len(p2p_nccl_data[rate][percentile]) > 2:
                p2p_nccl_simplified_ttft.append(p2p_nccl_data[rate][percentile][2])
            else:
                p2p_nccl_simplified_ttft.append(None)
                
            # Extract p2pnccl without kvtransfer TTFT data if available
            if len(p2p_nccl_data[rate][percentile]) > 3:
                p2p_nccl_no_kvtransfer_ttft.append(p2p_nccl_data[rate][percentile][3])
            else:
                p2p_nccl_no_kvtransfer_ttft.append(None)
    
    if not valid_rates:
        print(f"No common request rates found between the two configurations for {percentile}.")
        return
    
    # Convert ms to seconds
    simple_ttft_s = [ttft / 1000 for ttft in simple_ttft]
    simple_tpot_s = [tpot / 1000 for tpot in simple_tpot]
    simple_simplified_ttft_s = [ttft / 1000 if ttft is not None else None for ttft in simple_simplified_ttft]
    p2p_nccl_ttft_s = [ttft / 1000 for ttft in p2p_nccl_ttft]
    p2p_nccl_tpot_s = [tpot / 1000 for tpot in p2p_nccl_tpot]
    p2p_nccl_simplified_ttft_s = [ttft / 1000 if ttft is not None else None for ttft in p2p_nccl_simplified_ttft]
    p2p_nccl_no_kvtransfer_ttft_s = [ttft / 1000 if ttft is not None else None for ttft in p2p_nccl_no_kvtransfer_ttft]
    
    # Set up the plot with four subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 16))
    
    # Colors and markers as specified
    existing_color = 'blue'      # Existing systems
    prefill_color = 'orange'     # Prefill-only
    decode_color = 'green'       # Decode-only
    marker_style = 'o'           # Circular markers
    
    # Plot Simplified TTFT comparison (top subplot)
    ax1.plot(valid_rates, simple_ttft_s,
             color=existing_color, marker=marker_style, linewidth=2, markersize=6,
             label='Existing systems (TTFT)')
    ax1.plot(valid_rates, p2p_nccl_simplified_ttft_s,
             color=prefill_color, marker=marker_style, linewidth=2, markersize=6,
             label='Prefill-only (TTFT without KVTransfer and DecodeQueue)')
    ax1.set_xlabel('Rate(reqs/s)')
    ax1.set_ylabel(f'{percentile.upper()} TTFT/TTFT without KVTransfer and DecodeQueue(s)')
    ax1.set_title(f'{percentile.upper()} TTFT vs TTFT without KVTransfer and DecodeQueue Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(valid_rates)
    
    # Plot p2pnccl without kvtransfer TTFT vs simple TTFT comparison (second subplot)
    ax2.plot(valid_rates, simple_ttft_s,
             color=existing_color, marker=marker_style, linewidth=2, markersize=6,
             label='Existing systems (TTFT)')
    ax2.plot(valid_rates, p2p_nccl_no_kvtransfer_ttft_s,
             color=prefill_color, marker=marker_style, linewidth=2, markersize=6,
             label='Prefill-only (No KVTransfer)')
    ax2.set_xlabel('Rate(reqs/s)')
    ax2.set_ylabel(f'{percentile.upper()} TTFT(s)')
    ax2.set_title(f'{percentile.upper()} TTFT Comparison (Existing systems vs Prefill-only without KVTransfer)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(valid_rates)
    
    # Plot TTFT comparison (third subplot)
    ax3.plot(valid_rates, simple_ttft_s,
             color=existing_color, marker=marker_style, linewidth=2, markersize=6,
             label='Existing systems')
    ax3.plot(valid_rates, p2p_nccl_ttft_s,
             color=prefill_color, marker=marker_style, linewidth=2, markersize=6,
             label='Prefill-only')
    ax3.set_xlabel('Rate(reqs/s)')
    ax3.set_ylabel(f'{percentile.upper()} TTFT(s)')
    ax3.set_title(f'{percentile.upper()} TTFT Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(valid_rates)
    
    # Plot TPOT comparison (fourth subplot)
    ax4.plot(valid_rates, simple_tpot_s,
             color=existing_color, marker=marker_style, linewidth=2, markersize=6,
             label='Existing systems')
    ax4.plot(valid_rates, p2p_nccl_tpot_s,
             color=decode_color, marker=marker_style, linewidth=2, markersize=6,
             label='Decode-only')
    ax4.set_xlabel('Rate(reqs/s)')
    ax4.set_ylabel(f'{percentile.upper()} TPOT(s)')
    ax4.set_title(f'{percentile.upper()} TPOT Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(valid_rates)
    
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
        percentile_output_file = os.path.join(default_plots_dir, f"benchmark_comparison_{percentile}.png")
    
    plt.savefig(percentile_output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {percentile_output_file}")
    plt.close()  # Close the figure to free memory


def main():
    parser = argparse.ArgumentParser(description='Plot benchmark results comparison')
    parser.add_argument('simple_folder', help='Path to the simple configuration results folder')
    parser.add_argument('p2p_nccl_folder', help='Path to the p2p_nccl configuration results folder')
    parser.add_argument('--output', '-o',
                        help='Base output file path for the plots (PNG format). Default: ../plots/benchmark_comparison_*.png')
    
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
    print("Generating performance comparison plots for all percentiles...")
    plot_performance_comparison(simple_data, p2p_nccl_data, args.output)


if __name__ == '__main__':
    main()