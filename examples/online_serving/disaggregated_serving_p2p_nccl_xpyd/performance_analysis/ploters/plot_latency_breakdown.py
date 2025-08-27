#!/usr/bin/env python3
"""
Script to read benchmark results from JSON files and plot latency breakdown as stacked bar chart.

Usage:
    python plot_latency_breakdown.py <data_folder> [--output OUTPUT_FILE]
    
    Example:
        python plot_latency_breakdown.py ./p2p_nccl
        python plot_latency_breakdown.py ./p2p_nccl --output latency_breakdown.png
        
    Arguments:
        data_folder     Path to the data folder containing JSON benchmark results
        --output, -o    Output file path for the plot (PNG format)
                        Default: latency_breakdown.png
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
    Read all JSON files in a folder and extract latency breakdown data.
    
    Returns:
        dict: A dictionary with request_rate as keys and latency components as values
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
                
                # Extract required metrics
                mean_prefill_queue_time_ms = json_data.get('mean_prefill_queue_time_ms')
                mean_prefill_execute_time_ms = json_data.get('mean_prefill_execute_time_ms')
                mean_decode_queue_time_ms = json_data.get('mean_decode_queue_time_ms')
                mean_e2el_ms = json_data.get('mean_e2el_ms')
                mean_ttft_ms = json_data.get('mean_ttft_ms')
                
                # Check if all required metrics are present
                if None in [mean_prefill_queue_time_ms, mean_prefill_execute_time_ms, 
                            mean_decode_queue_time_ms, mean_e2el_ms, mean_ttft_ms]:
                    print(f"Warning: Missing required metrics in {filename}")
                    continue
                
                # Calculate decoding execution time
                mean_decode_execute_time_ms = mean_e2el_ms - mean_ttft_ms
                
                data[request_rate] = {
                    'mean_prefill_queue_time_ms': float(mean_prefill_queue_time_ms),
                    'mean_prefill_execute_time_ms': float(mean_prefill_execute_time_ms),
                    'mean_decode_queue_time_ms': float(mean_decode_queue_time_ms),
                    'mean_decode_execute_time_ms': float(mean_decode_execute_time_ms)
                }
                
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue
    
    return data


def plot_latency_breakdown(benchmark_data, output_file=None):
    """
    Plot latency breakdown as stacked bar chart.
    
    Args:
        benchmark_data (dict): Data from benchmark results
        output_file (str): Output file path for the plot
    """
    # Get all request rates and sort them
    all_rates = list(benchmark_data.keys())
    # Filter out infinite rates and sort
    finite_rates = [rate for rate in all_rates if rate != float('inf')]
    sorted_rates = sorted(finite_rates)
    
    if not sorted_rates:
        print("No valid request rates found.")
        return
    
    # Extract data for plotting
    per_gpu_rates = []
    prefill_queue_data = []
    prefill_execute_data = []
    decode_queue_data = []
    decode_execute_data = []
    
    for rate in sorted_rates:
        per_gpu_rate = rate / 2.0  # 1p1d configuration
        per_gpu_rates.append(per_gpu_rate)
        
        data = benchmark_data[rate]
        prefill_queue_data.append(data['mean_prefill_queue_time_ms'])
        prefill_execute_data.append(data['mean_prefill_execute_time_ms'])
        decode_queue_data.append(data['mean_decode_queue_time_ms'])
        decode_execute_data.append(data['mean_decode_execute_time_ms'])
    
    # Convert to percentages
    total_latency = (np.array(prefill_queue_data) + 
                     np.array(prefill_execute_data) + 
                     np.array(decode_queue_data) + 
                     np.array(decode_execute_data))
    
    prefill_queue_pct = np.array(prefill_queue_data) / total_latency * 100
    prefill_execute_pct = np.array(prefill_execute_data) / total_latency * 100
    decode_queue_pct = np.array(decode_queue_data) / total_latency * 100
    decode_execute_pct = np.array(decode_execute_data) / total_latency * 100
    
    # Set up the bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Colors as specified
    prefill_queue_color = 'lightgreen'    # Prefill Queuing
    prefill_execute_color = 'green'       # Prefill Execution
    decode_queue_color = 'skyblue'     # Decoding Queueing (deeper blue than skyblue)
    decode_execute_color = 'dodgerblue'    # Decoding Execution
    
    # Create stacked bar chart
    bar_width = 0.35
    x_positions = np.arange(len(per_gpu_rates))
    
    # Plot bars from bottom to top
    ax.bar(x_positions, prefill_queue_pct, bar_width, 
           label='Prefill Queuing', color=prefill_queue_color)
    ax.bar(x_positions, prefill_execute_pct, bar_width, 
           bottom=prefill_queue_pct, label='Prefill Execution', color=prefill_execute_color)
    ax.bar(x_positions, decode_queue_pct, bar_width, 
           bottom=prefill_queue_pct + prefill_execute_pct, 
           label='Decoding Queueing', color=decode_queue_color)
    ax.bar(x_positions, decode_execute_pct, bar_width, 
           bottom=prefill_queue_pct + prefill_execute_pct + decode_queue_pct, 
           label='Decoding Execution', color=decode_execute_color)
    
    # Set labels and title
    ax.set_xlabel('Per-GPU Rate (req/s)')
    ax.set_ylabel('Latency Breakdown (%)')
    ax.set_title('Latency Breakdown by Component')
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'{rate:.1f}' for rate in per_gpu_rates])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Use default output file name if none provided
    if not output_file:
        output_file = "latency_breakdown.png"
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot latency breakdown from benchmark results')
    parser.add_argument('data_folder', help='Path to the data folder containing JSON benchmark results')
    parser.add_argument('--output', '-o', default='latency_breakdown.png',
                        help='Output file path for the plot (PNG format). Default: latency_breakdown.png')
    
    args = parser.parse_args()
    
    # Read data from folder
    print("Reading benchmark data...")
    benchmark_data = read_benchmark_data(args.data_folder)
    print(f"Found {len(benchmark_data)} data points")
    
    if not benchmark_data:
        print("No data found. Exiting.")
        return
    
    # Plot the latency breakdown
    print("Generating latency breakdown plot...")
    plot_latency_breakdown(benchmark_data, args.output)


if __name__ == '__main__':
    main()