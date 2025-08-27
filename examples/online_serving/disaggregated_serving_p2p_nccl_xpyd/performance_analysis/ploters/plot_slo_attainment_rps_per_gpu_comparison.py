#!/usr/bin/env python3
"""
Script to read benchmark results from JSON files and plot SLO Attainment comparison
between simple and p2p_nccl configurations. SLO Attainment is the percentage of requests
that meet specific TTFT and TPOT limits.

Usage:
    python plot_slo_attainment_rps_per_gpu_comparison.py <simple_folder> <p2p_nccl_folder> [--output OUTPUT_FILE] [--ttft-limit TTFT_LIMIT] [--tpot-limit TPOT_LIMIT]
    
    Example:
        python plot_slo_attainment_rps_per_gpu_comparison.py ./simple ./p2p_nccl
        python plot_slo_attainment_rps_per_gpu_comparison.py ./simple ./p2p_nccl --output slo_comparison.png
        python plot_slo_attainment_rps_per_gpu_comparison.py ./simple ./p2p_nccl --ttft-limit 500 --tpot-limit 50
        
    Arguments:
        simple_folder     Path to the simple configuration results folder
        p2p_nccl_folder   Path to the p2p_nccl configuration results folder
        --output, -o      Output file path for the plot (PNG format)
                          Default: slo_attainment_rps_per_gpu_comparison.png
        --ttft-limit      TTFT limit in milliseconds (default: 400)
        --tpot-limit      TPOT limit in milliseconds (default: 40)
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


def calculate_slo_attainment(ttfts, tpots, ttft_limit, tpot_limit):
    """
    Calculate SLO attainment percentages for different criteria.
    
    Args:
        ttfts (list): List of TTFT values in milliseconds
        tpots (list): List of TPOT values in milliseconds
        ttft_limit (float): TTFT limit in milliseconds
        tpot_limit (float): TPOT limit in milliseconds
    
    Returns:
        tuple: (both_slo, ttft_only_slo, tpot_only_slo) - SLO attainment percentages
    """
    if not ttfts or not tpots or len(ttfts) != len(tpots):
        return 0.0, 0.0, 0.0
    
    total_requests = len(ttfts)
    both_met = 0
    ttft_met = 0
    tpot_met = 0
    
    for ttft, tpot in zip(ttfts, tpots):
        if ttft <= ttft_limit and tpot <= tpot_limit:
            both_met += 1
        if ttft <= ttft_limit:
            ttft_met += 1
        if tpot <= tpot_limit:
            tpot_met += 1
    
    both_slo = (both_met / total_requests) * 100
    ttft_only_slo = (ttft_met / total_requests) * 100
    tpot_only_slo = (tpot_met / total_requests) * 100
    
    return both_slo, ttft_only_slo, tpot_only_slo


def read_benchmark_data(folder_path):
    """
    Read all JSON files in a folder and extract per-request latency data for SLO calculation.
    
    Args:
        folder_path (str): Path to the folder containing JSON benchmark results
    
    Returns:
        dict: A dictionary with request_rate as keys and a dict of SLO attainment percentages as values.
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
                
                # Extract per-request latency data
                ttfts = json_data.get('ttfts', [])
                tpots = json_data.get('tpots', [])
                
                if not ttfts or not tpots:
                    print(f"Warning: Could not find per-request latency data in {filename}")
                    continue
                
                # Store the raw data for later SLO calculations with different limits
                data[request_rate] = {
                    'ttfts': ttfts,
                    'tpots': tpots
                }
                
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue
    
    return data


def plot_slo_attainment_comparison(simple_data, p2p_nccl_data, ttft_limit, tpot_limit, output_file=None):
    """
    Plot SLO attainment comparison between simple and p2p_nccl configurations.
    
    Args:
        simple_data (dict): Data from simple configuration
        p2p_nccl_data (dict): Data from p2p_nccl configuration
        ttft_limit (float): TTFT limit in milliseconds
        tpot_limit (float): TPOT limit in milliseconds
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
    simple_both_slo = []
    simple_ttft_slo = []
    simple_tpot_slo = []
    # For p2p_nccl configuration (will be adjusted for 1p1d)
    p2p_nccl_both_slo = []
    p2p_nccl_ttft_slo = []
    p2p_nccl_tpot_slo = []
    # Request rates that have data in both configurations
    valid_rates = []
    
    # Adjusted rates for p2p_nccl (divided by 2 for 1p1d configuration)
    p2p_nccl_adjusted_rates = []
    
    for rate in sorted_rates:
        # Check if rate exists in both datasets and has the required data
        if (rate in simple_data and rate in p2p_nccl_data and
            'ttfts' in simple_data[rate] and 'tpots' in simple_data[rate] and
            'ttfts' in p2p_nccl_data[rate] and 'tpots' in p2p_nccl_data[rate]):
            valid_rates.append(rate)
            # For p2p_nccl, adjust rate by dividing by 2 (1p1d configuration)
            p2p_nccl_adjusted_rates.append(rate / 2)
            
            # Calculate SLO attainment for simple configuration
            simple_ttfts = simple_data[rate]['ttfts']
            simple_tpots = simple_data[rate]['tpots']
            simple_both, simple_ttft_only, simple_tpot_only = calculate_slo_attainment(
                simple_ttfts, simple_tpots, ttft_limit, tpot_limit)
            simple_both_slo.append(simple_both)
            simple_ttft_slo.append(simple_ttft_only)
            simple_tpot_slo.append(simple_tpot_only)
            
            # Calculate SLO attainment for p2p_nccl configuration
            p2p_nccl_ttfts = p2p_nccl_data[rate]['ttfts']
            p2p_nccl_tpots = p2p_nccl_data[rate]['tpots']
            p2p_nccl_both, p2p_nccl_ttft_only, p2p_nccl_tpot_only = calculate_slo_attainment(
                p2p_nccl_ttfts, p2p_nccl_tpots, ttft_limit, tpot_limit)
            p2p_nccl_both_slo.append(p2p_nccl_both)
            p2p_nccl_ttft_slo.append(p2p_nccl_ttft_only)
            p2p_nccl_tpot_slo.append(p2p_nccl_tpot_only)
    
    if not valid_rates:
        print("No common request rates found between the two configurations.")
        return
    
    # Set up the plot with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    
    # Improved colors
    simple_color = '#1f77b4'  # Blue
    p2p_nccl_color = '#ff7f0e'  # Orange
    marker_style = 'o'  # Circular markers
    
    # Plot Both TTFT and TPOT SLO attainment (top subplot)
    ax1.plot(valid_rates, simple_both_slo,
             color=simple_color, marker=marker_style, linewidth=2, markersize=6,
             label='Simple')
    ax1.plot(p2p_nccl_adjusted_rates, p2p_nccl_both_slo,
             color=p2p_nccl_color, marker=marker_style, linewidth=2, markersize=6,
             label='P2P NCCL (1p1d)')
    ax1.set_xlabel('Request Rate / GPU (reqs/s/GPU)')
    ax1.set_ylabel('SLO Attainment (%)')
    ax1.set_title(f'SLO Attainment (TTFT ≤ {ttft_limit}ms AND TPOT ≤ {tpot_limit}ms)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # Plot TTFT-only SLO attainment (middle subplot)
    ax2.plot(valid_rates, simple_ttft_slo,
             color=simple_color, marker=marker_style, linewidth=2, markersize=6,
             label='Simple')
    ax2.plot(p2p_nccl_adjusted_rates, p2p_nccl_ttft_slo,
             color=p2p_nccl_color, marker=marker_style, linewidth=2, markersize=6,
             label='P2P NCCL (1p1d)')
    ax2.set_xlabel('Request Rate / GPU (reqs/s/GPU)')
    ax2.set_ylabel('SLO Attainment (%)')
    ax2.set_title(f'TTFT-only SLO Attainment (TTFT ≤ {ttft_limit}ms)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    # Plot TPOT-only SLO attainment (bottom subplot)
    ax3.plot(valid_rates, simple_tpot_slo,
             color=simple_color, marker=marker_style, linewidth=2, markersize=6,
             label='Simple')
    ax3.plot(p2p_nccl_adjusted_rates, p2p_nccl_tpot_slo,
             color=p2p_nccl_color, marker=marker_style, linewidth=2, markersize=6,
             label='P2P NCCL (1p1d)')
    ax3.set_xlabel('Request Rate / GPU (reqs/s/GPU)')
    ax3.set_ylabel('SLO Attainment (%)')
    ax3.set_title(f'TPOT-only SLO Attainment (TPOT ≤ {tpot_limit}ms)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 100)
    
    plt.tight_layout()
    
    # Determine output file name
    if output_file:
        output_filename = output_file
    else:
        output_filename = 'slo_attainment_rps_per_gpu_comparison.png'
    
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_filename}")
    plt.close()  # Close the figure to free memory


def main():
    parser = argparse.ArgumentParser(description='Plot SLO attainment comparison with RPS/GPU')
    parser.add_argument('simple_folder', help='Path to the simple configuration results folder')
    parser.add_argument('p2p_nccl_folder', help='Path to the p2p_nccl configuration results folder')
    parser.add_argument('--output', '-o', default='slo_attainment_rps_per_gpu_comparison.png',
                        help='Output file path for the plot (PNG format). Default: slo_attainment_rps_per_gpu_comparison.png')
    parser.add_argument('--ttft-limit', type=float, default=400.0,
                        help='TTFT limit in milliseconds (default: 400)')
    parser.add_argument('--tpot-limit', type=float, default=40.0,
                        help='TPOT limit in milliseconds (default: 40)')
    
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
    
    # Plot the SLO attainment comparison
    print(f"Generating SLO attainment comparison plots with TTFT limit={args.ttft_limit}ms, TPOT limit={args.tpot_limit}ms...")
    plot_slo_attainment_comparison(simple_data, p2p_nccl_data, args.ttft_limit, args.tpot_limit, args.output)


if __name__ == '__main__':
    main()