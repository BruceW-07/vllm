#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Script to read benchmark results from JSON files and plot latency breakdown as stacked bar chart.

Usage:
    python plot_latency_breakdown.py <data_folder> [--output OUTPUT_FILE] [--num-gpus NUM_GPUS]

    Example:
        python plot_latency_breakdown.py ./p2p_nccl
        python plot_latency_breakdown.py ./p2p_nccl --output latency_breakdown.png
        python plot_latency_breakdown.py ./p2p_nccl --plot-type latency
        python plot_latency_breakdown.py ./p2p_nccl --plot-type ttft
        python plot_latency_breakdown.py ./p2p_nccl --num-gpus 2

    Arguments:
        data_folder     Path to the data folder containing JSON benchmark results
        --output, -o    Output file path for the plot (PNG format)
                        Default: latency_breakdown.png
        --plot-type, -p Type of plot to generate (latency, ttft, or both)
                        Default: both
        --num-gpus      Number of GPUs used in the configuration
                        Default: 1
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def extract_request_rate_from_filename(filename):
    """Extract request rate from filename."""
    # Assuming filename format like: label-request_rateqps-model-date.json
    parts = filename.split("-")
    for part in parts:
        if "qps" in part:
            # Extract the numeric part before 'qps'
            rate_str = part.replace("qps", "")
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
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)

            try:
                with open(file_path) as f:
                    json_data = json.load(f)

                # Try to get request_rate from JSON data first
                request_rate = None
                if "request_rate" in json_data:
                    request_rate = json_data["request_rate"]
                    # Convert "inf" to a large number for sorting purposes
                    if request_rate == "inf":
                        request_rate = float("inf")
                    else:
                        request_rate = float(request_rate)

                # If not in JSON data, try to extract from filename
                if request_rate is None:
                    request_rate = extract_request_rate_from_filename(filename)

                if request_rate is None:
                    print(f"Warning: Could not determine request rate for {filename}")
                    continue

                # Extract required metrics
                mean_prefill_queue_time_ms = json_data.get("mean_prefill_queue_time_ms")
                mean_prefill_execute_time_ms = json_data.get(
                    "mean_prefill_execute_time_ms"
                )
                mean_decode_queue_time_ms = json_data.get("mean_decode_queue_time_ms")
                mean_e2el_ms = json_data.get("mean_e2el_ms")
                mean_ttft_ms = json_data.get("mean_ttft_ms")
                mean_kv_load_time_ms = json_data.get("mean_kv_load_time_ms")
                mean_kv_save_time_ms = json_data.get("mean_kv_save_time_ms")
                mean_decode_execute_time_ms = json_data.get(
                    "mean_decode_execute_time_ms"
                )  # First token decode time

                # Check if all required metrics are present
                if None in [
                    mean_prefill_queue_time_ms,
                    mean_prefill_execute_time_ms,
                    mean_decode_queue_time_ms,
                    mean_e2el_ms,
                    mean_ttft_ms,
                ]:
                    print(f"Warning: Missing required metrics in {filename}")
                    continue

                # Calculate total decoding execution time (all tokens)
                mean_total_decode_execute_time_ms = mean_e2el_ms - mean_ttft_ms

                data[request_rate] = {
                    "mean_prefill_queue_time_ms": float(mean_prefill_queue_time_ms),
                    "mean_prefill_execute_time_ms": float(mean_prefill_execute_time_ms),
                    "mean_decode_queue_time_ms": float(mean_decode_queue_time_ms),
                    "mean_first_token_decode_time_ms": float(
                        mean_decode_execute_time_ms
                    ),  # First token decode time
                    "mean_total_decode_execute_time_ms": float(
                        mean_total_decode_execute_time_ms
                    ),  # All tokens decode time
                    "mean_kv_load_time_ms": float(mean_kv_load_time_ms)
                    if mean_kv_load_time_ms is not None
                    else 0.0,
                    "mean_kv_save_time_ms": float(mean_kv_save_time_ms)
                    if mean_kv_save_time_ms is not None
                    else 0.0,
                    "mean_ttft_ms": float(mean_ttft_ms),
                }

            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue

    return data


def plot_latency_breakdown(benchmark_data, output_file, num_gpus=1):
    """
    Plot E2E latency breakdown as stacked bar chart.
    E2E latency is simplified into two components: TTFT and remaining tokens decoding time.

    Args:
        benchmark_data (dict): Data from benchmark results
        output_file (str): Output file path for the plot
        num_gpus (int): Number of GPUs used in the configuration
    """
    # Get all request rates and sort them
    all_rates = list(benchmark_data.keys())
    # Filter out infinite rates and sort
    finite_rates = [rate for rate in all_rates if rate != float("inf")]
    sorted_rates = sorted(finite_rates)

    if not sorted_rates:
        print("No valid request rates found.")
        return

    # Extract data for plotting
    per_gpu_rates = []
    ttft_data = []
    remaining_decode_data = []

    for rate in sorted_rates:
        per_gpu_rate = rate / num_gpus
        per_gpu_rates.append(per_gpu_rate)

        data = benchmark_data[rate]
        ttft_data.append(data["mean_ttft_ms"])
        remaining_decode_data.append(data["mean_total_decode_execute_time_ms"])

    # Convert to percentages
    total_latency = np.array(ttft_data) + np.array(remaining_decode_data)

    ttft_pct = np.array(ttft_data) / total_latency * 100
    remaining_decode_pct = np.array(remaining_decode_data) / total_latency * 100

    # Set up the bar chart
    fig, ax = plt.subplots(figsize=(12, 8))

    # Colors for the two main components
    ttft_color = "lightblue"  # TTFT
    remaining_decode_color = "darkblue"  # Remaining Tokens Decoding

    # Create stacked bar chart
    bar_width = 0.35
    x_positions = np.arange(len(per_gpu_rates))

    # Plot bars from bottom to top
    ax.bar(
        x_positions,
        ttft_pct,
        bar_width,
        label="TTFT (Time to First Token)",
        color=ttft_color,
    )
    ax.bar(
        x_positions,
        remaining_decode_pct,
        bar_width,
        bottom=ttft_pct,
        label="Remaining Tokens Decoding",
        color=remaining_decode_color,
    )

    # Set labels and title
    ax.set_xlabel("Per-GPU Rate (req/s)")
    ax.set_ylabel("E2E Latency Breakdown (%)")
    ax.set_title("E2E Latency Breakdown: TTFT vs Remaining Tokens Decoding")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{rate:.1f}" for rate in per_gpu_rates])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {output_file}")


def plot_ttft_breakdown(benchmark_data, output_file, num_gpus=1):
    """
    Plot TTFT breakdown as stacked bar chart.
    TTFT includes: prefill_queue, prefill_execute, kv_load, kv_save, decode_queue, decode_execute.
    Also shows unknown portion if the sum doesn't equal TTFT.
    Reports error if component sum exceeds actual TTFT but continues plotting.

    Args:
        benchmark_data (dict): Data from benchmark results
        output_file (str): Output file path for the plot
        num_gpus (int): Number of GPUs used in the configuration
    """
    # Get all request rates and sort them
    all_rates = list(benchmark_data.keys())
    # Filter out infinite rates and sort
    finite_rates = [rate for rate in all_rates if rate != float("inf")]
    sorted_rates = sorted(finite_rates)

    if not sorted_rates:
        print("No valid request rates found.")
        return

    # Extract data for plotting
    per_gpu_rates = []
    prefill_queue_data = []
    prefill_execute_data = []
    kv_load_data = []
    kv_save_data = []
    decode_queue_data = []
    first_token_decode_data = []
    ttft_data = []

    for rate in sorted_rates:
        per_gpu_rate = rate / num_gpus
        per_gpu_rates.append(per_gpu_rate)

        data = benchmark_data[rate]
        prefill_queue_data.append(data["mean_prefill_queue_time_ms"])
        prefill_execute_data.append(data["mean_prefill_execute_time_ms"])
        # Separate KV load and save times
        kv_load = data.get("mean_kv_load_time_ms", 0)
        kv_save = data.get("mean_kv_save_time_ms", 0)
        kv_load_data.append(kv_load)
        kv_save_data.append(kv_save)
        decode_queue_data.append(data["mean_decode_queue_time_ms"])
        first_token_decode_data.append(data["mean_first_token_decode_time_ms"])
        # Calculate actual TTFT from components for comparison
        ttft_data.append(
            data["mean_prefill_queue_time_ms"]
            + data["mean_prefill_execute_time_ms"]
            + kv_load
            + kv_save
            + data["mean_decode_queue_time_ms"]
            + data["mean_first_token_decode_time_ms"]
        )

    # Calculate unknown portion (difference between actual TTFT and sum of components)
    actual_ttft = np.array(
        [benchmark_data[rate]["mean_ttft_ms"] for rate in sorted_rates]
    )
    component_sum = (
        np.array(prefill_queue_data)
        + np.array(prefill_execute_data)
        + np.array(kv_load_data)
        + np.array(kv_save_data)
        + np.array(decode_queue_data)
        + np.array(first_token_decode_data)
    )
    
    # Check if component sum is greater than actual TTFT and report error
    diff = component_sum - actual_ttft
    if np.any(diff > 0.001):  # Allow small numerical errors
        max_diff_idx = np.argmax(diff)
        print(f"ERROR: Component sum exceeds actual TTFT!")
        print(f"Worst case at rate {sorted_rates[max_diff_idx]}: "
              f"component sum = {component_sum[max_diff_idx]:.3f}ms, "
              f"actual TTFT = {actual_ttft[max_diff_idx]:.3f}ms, "
              f"difference = {diff[max_diff_idx]:.3f}ms")
        print("Continuing with plot generation...")
    
    unknown_data = np.maximum(actual_ttft - component_sum, 0)  # Ensure non-negative

    # Convert to percentages of actual TTFT
    ttft_array = np.array(actual_ttft)
    prefill_queue_pct = np.array(prefill_queue_data) / ttft_array * 100
    prefill_execute_pct = np.array(prefill_execute_data) / ttft_array * 100
    kv_load_pct = np.array(kv_load_data) / ttft_array * 100
    kv_save_pct = np.array(kv_save_data) / ttft_array * 100
    decode_queue_pct = np.array(decode_queue_data) / ttft_array * 100
    first_token_decode_pct = np.array(first_token_decode_data) / ttft_array * 100
    unknown_pct = np.array(unknown_data) / ttft_array * 100

    # Set up the bar chart
    fig, ax = plt.subplots(figsize=(12, 8))

    # Colors for each component
    prefill_queue_color = "lightgreen"  # Prefill Queuing
    prefill_execute_color = "green"  # Prefill Execution
    kv_load_color = "orange"  # KV Load
    kv_save_color = "darkorange"  # KV Save
    decode_queue_color = "skyblue"  # Decoding Queueing
    first_token_decode_color = "dodgerblue"  # First Token Decoding
    unknown_color = "gray"  # Unknown

    # Create stacked bar chart
    bar_width = 0.35
    x_positions = np.arange(len(per_gpu_rates))

    # Plot bars from bottom to top
    bottom = np.zeros(len(per_gpu_rates))

    ax.bar(
        x_positions,
        prefill_queue_pct,
        bar_width,
        label="Prefill Queuing",
        color=prefill_queue_color,
        bottom=bottom,
    )
    bottom += prefill_queue_pct

    ax.bar(
        x_positions,
        prefill_execute_pct,
        bar_width,
        label="Prefill Execution",
        color=prefill_execute_color,
        bottom=bottom,
    )
    bottom += prefill_execute_pct

    ax.bar(
        x_positions,
        kv_load_pct,
        bar_width,
        label="KV Load",
        color=kv_load_color,
        bottom=bottom,
    )
    bottom += kv_load_pct

    ax.bar(
        x_positions,
        kv_save_pct,
        bar_width,
        label="KV Save",
        color=kv_save_color,
        bottom=bottom,
    )
    bottom += kv_save_pct

    ax.bar(
        x_positions,
        decode_queue_pct,
        bar_width,
        label="Decoding Queueing",
        color=decode_queue_color,
        bottom=bottom,
    )
    bottom += decode_queue_pct

    ax.bar(
        x_positions,
        first_token_decode_pct,
        bar_width,
        label="First Token Decoding",
        color=first_token_decode_color,
        bottom=bottom,
    )
    bottom += first_token_decode_pct

    ax.bar(
        x_positions,
        unknown_pct,
        bar_width,
        label="Unknown",
        color=unknown_color,
        bottom=bottom,
    )

    # Set labels and title
    ax.set_xlabel("Per-GPU Rate (req/s)")
    ax.set_ylabel("TTFT Breakdown (%)")
    ax.set_title("TTFT Breakdown by Component")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{rate:.1f}" for rate in per_gpu_rates])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"TTFT breakdown plot saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot latency breakdown from benchmark results"
    )
    parser.add_argument(
        "data_folder", help="Path to the data folder containing JSON benchmark results"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path for the plot (PNG format). Default: ../plots/latency_breakdown.png or ../plots/ttft_breakdown.png",
    )
    parser.add_argument(
        "--plot-type",
        "-p",
        default="both",
        choices=["latency", "ttft", "both"],
        help="Type of plot to generate. Default: both",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs used in the configuration. Default: 1",
    )

    args = parser.parse_args()

    # Read data from folder
    print("Reading benchmark data...")
    benchmark_data = read_benchmark_data(args.data_folder)
    print(f"Found {len(benchmark_data)} data points")

    if not benchmark_data:
        print("No data found. Exiting.")
        return

    # Plot the selected breakdown
    # Handle latency breakdown
    if args.plot_type in ["latency", "both"]:
        print("Generating latency breakdown plot...")
        # Handle output file path
        output_file = args.output
        if not output_file:
            # Create default plots directory if it doesn't exist
            default_plots_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", "plots"
            )
            os.makedirs(default_plots_dir, exist_ok=True)
            output_file = os.path.join(default_plots_dir, "latency_breakdown.png")
        plot_latency_breakdown(benchmark_data, output_file, args.num_gpus)

    # Handle TTFT breakdown
    if args.plot_type in ["ttft", "both"]:
        print("Generating TTFT breakdown plot...")
        # Handle output file path
        output_file = args.output
        if not output_file or "latency_breakdown" in output_file:
            # Create default plots directory if it doesn't exist
            default_plots_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", "plots"
            )
            os.makedirs(default_plots_dir, exist_ok=True)
            output_file = os.path.join(default_plots_dir, "ttft_breakdown.png")
        plot_ttft_breakdown(benchmark_data, output_file, args.num_gpus)


if __name__ == "__main__":
    main()
