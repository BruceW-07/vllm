#!/usr/bin/env python3
"""
vLLM Benchmark Result Plotting Script
类似于 DistServe _plot.py 的功能，用于绘制 vLLM benchmark 结果
"""

import argparse
import dataclasses
import json
import os
import re
import sys
from typing import List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print(
        "Warning: matplotlib and/or numpy not available. Plotting functions disabled."
    )


@dataclasses.dataclass
class Backend:
    name: str
    label: str
    color: str
    marker: str = "o"


def load_vllm_result(result_file: str) -> dict:
    """Load vLLM benchmark result from JSON file"""
    if not os.path.exists(result_file):
        raise FileNotFoundError(f"Result file not found: {result_file}")

    with open(result_file) as f:
        data = json.load(f)

    return data


def get_attainment(ttfts: List[float], tpots: List[float],
                   ttft_slo: Optional[float],
                   tpot_slo: Optional[float]) -> float:
    """Calculate SLO attainment rate for given TTFT and TPOT data"""
    if ttft_slo is None:
        ttft_slo = 1e10
    if tpot_slo is None:
        tpot_slo = 1e10

    if len(ttfts) != len(tpots):
        raise ValueError(
            f"TTFT and TPOT arrays must have same length: {len(ttfts)} vs {len(tpots)}"
        )

    counter = 0
    for ttft, tpot in zip(ttfts, tpots):
        if ttft <= ttft_slo and tpot <= tpot_slo:
            counter += 1

    return (counter / len(ttfts)) * 100


def find_intersection(xs: List[float], ys: List[float],
                      target_y: float) -> Tuple[Optional[float], Optional[float]]:
    """Find intersection point of curve with horizontal line at target_y"""
    for index in range(len(xs) - 1):
        x0, x1 = xs[index], xs[index + 1]
        y0, y1 = ys[index], ys[index + 1]

        if (y0 < target_y) != (y1 < target_y):
            # Intersection found
            inter_x = (target_y - y0) * (x1 - x0) / (y1 - y0) + x0
            return (inter_x, target_y)

    # No intersection found
    return (None, None)


def draw_attainment_rate_plot(
        ax,  # plt.Axes when matplotlib available
        result_files: List[str],  # List of vLLM result JSON files
        request_rates: List[
            float],  # Corresponding request rates for each file
        backends: List[Backend],
        ttft_slo: float,  # TTFT SLO in ms
        tpot_slo: float,  # TPOT SLO in ms
        atta_target: Optional[float] = None,
        show_ylabel: bool = False):
    """Draw attainment rate plot similar to DistServe's version"""
    if not HAS_PLOTTING:
        print("Plotting not available - matplotlib/numpy missing")
        return

    ax.set_xlabel("Request Rate (req/s)")
    if show_ylabel:
        ax.set_ylabel("SLO Attainment (%)")

    if len(result_files) != len(request_rates):
        raise ValueError(
            "Number of result files must match number of request rates")

    first_inter_x = -1
    for backend in backends:
        xs = request_rates
        ys_both = []
        ys_ttft = []
        ys_tpot = []

        for result_file in result_files:
            try:
                data = load_vllm_result(result_file)

                # Extract TTFT and TPOT data
                ttfts = [t * 1000
                         for t in data.get('ttfts', [])]  # Convert to ms
                tpots = [t * 1000
                         for t in data.get('tpots', [])]  # Convert to ms

                if not ttfts or not tpots:
                    print(f"WARNING: No TTFT/TPOT data found in {result_file}")
                    ys_both.append(0)
                    ys_ttft.append(0)
                    ys_tpot.append(0)
                    continue

                # Calculate attainment rates
                ys_both.append(get_attainment(ttfts, tpots, ttft_slo,
                                              tpot_slo))
                ys_ttft.append(get_attainment(ttfts, tpots, ttft_slo, None))
                ys_tpot.append(get_attainment(ttfts, tpots, None, tpot_slo))

            except Exception as e:
                print(f"Error processing {result_file}: {e}")
                ys_both.append(0)
                ys_ttft.append(0)
                ys_tpot.append(0)

        # Plot lines with different colors and markers
        ax.plot(xs,
                ys_both,
                label="Both TTFT & TPOT",
                color="C0",
                marker="o",
                linewidth=2)
        ax.plot(xs,
                ys_ttft,
                label="TTFT only",
                linestyle=":",
                color="C1",
                marker="s",
                linewidth=2)
        ax.plot(xs,
                ys_tpot,
                label="TPOT only",
                linestyle="--",
                color="C2",
                marker="^",
                linewidth=2)

        if atta_target:
            # Draw target line
            ax.axhline(y=atta_target, color="grey", linestyle="--", alpha=0.7)
            
            # Check and draw intersection lines for each curve
            try:
                inter_x, inter_y = find_intersection(xs, ys_both, atta_target)
                if inter_x is not None:
                    ax.vlines(x=inter_x, ymin=0, ymax=inter_y,
                             linestyles="--", colors="C0", alpha=0.8)
            except:
                pass
                
            try:
                inter_x_ttft, inter_y_ttft = find_intersection(xs, ys_ttft, atta_target)
                if inter_x_ttft is not None:
                    ax.vlines(x=inter_x_ttft, ymin=0, ymax=inter_y_ttft,
                             linestyles="--", colors="C1", alpha=0.8)
            except:
                pass
                
            try:
                inter_x_tpot, inter_y_tpot = find_intersection(xs, ys_tpot, atta_target)
                if inter_x_tpot is not None:
                    ax.vlines(x=inter_x_tpot, ymin=0, ymax=inter_y_tpot,
                             linestyles="--", colors="C2", alpha=0.8)
            except:
                pass
                
            if first_inter_x == -1:
                try:
                    first_inter_x, _ = find_intersection(xs, ys_both, atta_target)
                except:
                    pass
            else:
                try:
                    current_inter_x, _ = find_intersection(xs, ys_both, atta_target)
                    if current_inter_x is not None and first_inter_x is not None:
                        print(
                            f"Improvement ({backends[0].label} compared to {backend.label}): {first_inter_x/current_inter_x}"
                        )
                except:
                    pass

    ax.set_ylim(0, 105)


def draw_slo_scale_plot(
        ax,  # plt.Axes when matplotlib available
        result_file: str,  # Single vLLM result JSON file
        backend: Backend,
        ttft_slo: float,  # Base TTFT SLO in ms
        tpot_slo: float,  # Base TPOT SLO in ms
        scales: List[float],  # SLO scale factors
        atta_target: Optional[float] = None,
        show_ylabel: bool = False):
    """Draw SLO scale plot similar to DistServe's version"""
    if not HAS_PLOTTING:
        print("Plotting not available - matplotlib/numpy missing")
        return

    ax.set_xlabel("SLO Scale")
    if show_ylabel:
        ax.set_ylabel("SLO Attainment (%)")

    scales = sorted(scales, reverse=True)
    ax.invert_xaxis()
    ax.set_ylim(0, 105)

    try:
        data = load_vllm_result(result_file)

        # Extract TTFT and TPOT data
        ttfts = [t * 1000 for t in data.get('ttfts', [])]  # Convert to ms
        tpots = [t * 1000 for t in data.get('tpots', [])]  # Convert to ms

        if not ttfts or not tpots:
            print(f"WARNING: No TTFT/TPOT data found in {result_file}")
            return

        xs = []
        ys_both = []
        ys_ttft = []
        ys_tpot = []

        for scale in scales:
            xs.append(scale)
            ys_both.append(
                get_attainment(ttfts, tpots, ttft_slo * scale,
                               tpot_slo * scale))
            ys_ttft.append(get_attainment(ttfts, tpots, ttft_slo * scale,
                                          None))
            ys_tpot.append(get_attainment(ttfts, tpots, None,
                                          tpot_slo * scale))

        # Plot lines with different colors and markers
        ax.plot(xs,
                ys_both,
                label="Both TTFT & TPOT",
                color="C0",
                marker="o",
                linewidth=2)
        ax.plot(xs,
                ys_ttft,
                label="TTFT only",
                linestyle=":",
                color="C1",
                marker="s",
                linewidth=2)
        ax.plot(xs,
                ys_tpot,
                label="TPOT only",
                linestyle="--",
                color="C2",
                marker="^",
                linewidth=2)

        if atta_target:
            # Draw target line
            ax.axhline(y=atta_target, color="grey", linestyle="--", alpha=0.7)
            
            # Check and draw intersection lines for each curve
            try:
                inter_x, inter_y = find_intersection(xs, ys_both, atta_target)
                if inter_x is not None:
                    ax.vlines(x=inter_x, ymin=0, ymax=inter_y,
                             linestyles="--", colors="C0", alpha=0.8)
            except:
                pass
                
            try:
                inter_x_ttft, inter_y_ttft = find_intersection(xs, ys_ttft, atta_target)
                if inter_x_ttft is not None:
                    ax.vlines(x=inter_x_ttft, ymin=0, ymax=inter_y_ttft,
                             linestyles="--", colors="C1", alpha=0.8)
            except:
                pass
                
            try:
                inter_x_tpot, inter_y_tpot = find_intersection(xs, ys_tpot, atta_target)
                if inter_x_tpot is not None:
                    ax.vlines(x=inter_x_tpot, ymin=0, ymax=inter_y_tpot,
                             linestyles="--", colors="C2", alpha=0.8)
            except:
                pass

    except Exception as e:
        print(f"Error processing {result_file}: {e}")


def extract_qps_from_filename(filename: str) -> Optional[float]:
    """Extract QPS value from vLLM result filename"""
    import re

    # Match patterns like "vllm-2.5qps-", "openai-1.0qps-", etc.
    match = re.search(r'-(\d+(?:\.\d+)?)qps-', filename)
    if match:
        return float(match.group(1))

    # Also try to match "inf" for infinite rate
    if 'infqps' in filename:
        return float('inf')

    return None


def scan_result_files(directory: str) -> List[Tuple[str, float]]:
    """Scan directory for vLLM result files and extract QPS values"""
    files_and_rates = []

    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return files_and_rates

    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            qps = extract_qps_from_filename(filename)
            if qps is not None:
                files_and_rates.append((filepath, qps))

    # Sort by QPS value
    files_and_rates.sort(key=lambda x: x[1])
    return files_and_rates


def plot_vllm_fig9_style(result_dir: str = ".",
                         ttft_slo: float = 125.0,
                         tpot_slo: float = 200.0,
                         atta_target: float = 90.0):
    """Create a plot similar to DistServe's plot_fig9
    
    Args:
        result_dir: Directory containing vLLM result files
        ttft_slo: TTFT SLO threshold in milliseconds (default: 125ms)
        tpot_slo: TPOT SLO threshold in milliseconds (default: 200ms)
        atta_target: Target attainment rate percentage (default: 90%)
    """
    if not HAS_PLOTTING:
        print("Cannot create plots - matplotlib/numpy not available")
        return

    plt.rcParams.update({'font.size': 16})
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    # Scan directory for result files
    files_and_rates = scan_result_files(result_dir)

    if files_and_rates:
        print(f"Found {len(files_and_rates)} result files in {result_dir}")
        
        # Build comprehensive title with configuration info
        title_info = ""
        try:
            first_file = files_and_rates[0][0]
            data = load_vllm_result(first_file)
            
            # Extract configuration for title
            title_parts = []
            if 'model' in data:
                title_parts.append(f"Model:{data['model']}")
            if 'num_prefill_instances' in data and 'num_decode_instances' in data:
                title_parts.append(f"PF:{data['num_prefill_instances']}, D:{data['num_decode_instances']}")
            if 'prefiller_tp_size' in data and 'decoder_tp_size' in data:
                title_parts.append(f"TP:{data['prefiller_tp_size']}x{data['decoder_tp_size']}")
            if 'dataset_name' in data:
                title_parts.append(f"Dataset:{data['dataset_name']}")
            
            if title_parts:
                title_info = f"vLLM Benchmark Results ({', '.join(title_parts)})"
            else:
                title_info = "vLLM Benchmark Results"
                
        except Exception as e:
            title_info = "vLLM Benchmark Results"
            print(f"Could not extract config for title: {e}")

        for filepath, qps in files_and_rates:
            print(f"  {os.path.basename(filepath)} -> {qps} QPS")

        result_files = [f for f, r in files_and_rates]
        request_rates = [r for f, r in files_and_rates]

        # First subplot: Attainment rate plot
        draw_attainment_rate_plot(axs[0],
                                  result_files,
                                  request_rates,
                                  [Backend("vllm", "vLLM", "C1")],
                                  ttft_slo=ttft_slo,
                                  tpot_slo=tpot_slo,
                                  atta_target=atta_target,
                                  show_ylabel=True)
        axs[0].set_title("SLO Attainment vs Request Rate")

        # Second subplot: SLO scale plot using middle file
        middle_idx = len(result_files) // 2
        draw_slo_scale_plot(axs[1],
                            result_files[middle_idx],
                            Backend("vllm", "vLLM", "C1"),
                            ttft_slo=ttft_slo,
                            tpot_slo=tpot_slo,
                            scales=[
                                1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7,
                                0.6, 0.5, 0.4
                            ],
                            atta_target=atta_target,
                            show_ylabel=True)
        axs[1].set_title("SLO Attainment vs SLO Scale")

        # Add overall title (closer to subplots)
        fig.suptitle(title_info, fontsize=18, y=0.92)

        # Add simplified legend centered at the bottom (closer to subplots)
        handles, labels = axs[0].get_legend_handles_labels()
        # Only show unique labels (remove duplicates from second subplot)
        unique_labels = []
        unique_handles = []
        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_labels.append(label)
                unique_handles.append(handle)
        
        fig.legend(unique_handles, unique_labels, 
                   loc='lower center', 
                   bbox_to_anchor=(0.5, 0.05),
                   ncol=3, 
                   frameon=False,
                   fontsize=14)
    else:
        print(f"No result files found in {result_dir}")
        return

    # Adjust layout to accommodate title and legend with more space for plots
    plt.subplots_adjust(top=0.85, bottom=0.18)
    plt.tight_layout(rect=[0, 0.15, 1, 0.87])

    # Check if output directory is specified via environment variable
    custom_output_dir = os.environ.get('PLOT_OUTPUT_DIR')
    
    if custom_output_dir:
        # Use the specified directory
        plots_dir = custom_output_dir
        os.makedirs(plots_dir, exist_ok=True)
    else:
        # Create plots directory if it doesn't exist (default behavior)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        plots_dir = os.path.join(script_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

    output_path = os.path.join(plots_dir, "vllm_benchmark_plots.pdf")
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    # Don't show plot in headless environment
    # plt.show()


def plot_custom(files_and_rates: List[Tuple[str, float]],
                ttft_slo: float,
                tpot_slo: float,
                output_file: str = "custom_plots.pdf"):
    """
    Create custom plots with specified files and SLO values
    
    Args:
        files_and_rates: List of (filename, request_rate) tuples
        ttft_slo: TTFT SLO threshold in ms
        tpot_slo: TPOT SLO threshold in ms
        output_file: Output PDF filename (will be saved in plots/ subdirectory)
    """
    if not HAS_PLOTTING:
        print("Cannot create plots - matplotlib/numpy not available")
        return

    plt.rcParams.update({'font.size': 14})
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    result_files = [f for f, r in files_and_rates]
    request_rates = [r for f, r in files_and_rates]

    # Attainment rate plot
    draw_attainment_rate_plot(axs[0],
                              result_files,
                              request_rates, [Backend("vllm", "vLLM", "C1")],
                              ttft_slo=ttft_slo,
                              tpot_slo=tpot_slo,
                              atta_target=90,
                              show_ylabel=True)
    axs[0].set_title("SLO Attainment vs Request Rate")

    # SLO scale plot using the last file
    if result_files:
        draw_slo_scale_plot(
            axs[1],
            result_files[-1],
            Backend("vllm", "vLLM", "C1"),
            ttft_slo=ttft_slo,
            tpot_slo=tpot_slo,
            scales=[2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 0.6, 0.4],
            atta_target=90,
            show_ylabel=True)
        axs[1].set_title("SLO Attainment vs SLO Scale")

    # Add simplified legend for custom plots
    handles, labels = axs[0].get_legend_handles_labels()
    # Only show unique labels 
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    
    fig.legend(unique_handles, unique_labels, 
               loc='lower center', 
               bbox_to_anchor=(0.5, 0.05),
               ncol=3, 
               frameon=False,
               fontsize=12)

    # Proper layout adjustment with more space for plots
    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout(rect=[0, 0.12, 1, 1])

    # Check if output directory is specified via environment variable
    custom_output_dir = os.environ.get('PLOT_OUTPUT_DIR')
    
    if custom_output_dir:
        # Use the specified directory
        plots_dir = custom_output_dir
        os.makedirs(plots_dir, exist_ok=True)
    else:
        # Create plots directory if it doesn't exist (default behavior)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        plots_dir = os.path.join(script_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

    output_path = os.path.join(plots_dir, output_file)
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Plots saved to {output_path}")
    # Don't show plot in headless environment
    # plt.show()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="vLLM Benchmark Result Plotting Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default parameters
  python vllm_plot.py plot --dir /path/to/results
  
  # Custom SLO thresholds
  python vllm_plot.py plot --dir /path/to/results --ttft-slo 100 --tpot-slo 150 --target 95
  
  # Custom plots with manual file specification
  python vllm_plot.py custom --ttft-slo 125 --tpot-slo 200 --files result1.json:2.0 result2.json:4.0

Notes:
- Use vLLM serve.py with --save-result --save-detailed to generate result files
- TTFT/TPOT SLO values are in milliseconds
- Result files should contain 'ttfts' and 'tpots' arrays for detailed analysis
- File names should contain QPS values like: 'vllm-2.5qps-model-date.json'
        """)

    subparsers = parser.add_subparsers(dest='command',
                                       help='Available commands')

    # Plot command
    plot_parser = subparsers.add_parser(
        'plot', help='Generate plots from result files')
    plot_parser.add_argument(
        '--dir',
        '-d',
        default='.',
        help='Directory containing result files (default: current directory)')
    plot_parser.add_argument(
        '--ttft-slo',
        type=float,
        default=125.0,
        help='TTFT SLO threshold in milliseconds (default: 125)')
    plot_parser.add_argument(
        '--tpot-slo',
        type=float,
        default=200.0,
        help='TPOT SLO threshold in milliseconds (default: 200)')
    plot_parser.add_argument(
        '--target',
        type=float,
        default=90.0,
        help='Target attainment rate percentage (default: 90)')

    # Custom command
    custom_parser = subparsers.add_parser(
        'custom', help='Create custom plots with specified files')
    custom_parser.add_argument('--ttft-slo',
                               type=float,
                               required=True,
                               help='TTFT SLO threshold in milliseconds')
    custom_parser.add_argument('--tpot-slo',
                               type=float,
                               required=True,
                               help='TPOT SLO threshold in milliseconds')
    custom_parser.add_argument(
        '--files',
        nargs='+',
        required=True,
        help='Files with rates in format: file1.json:rate1 file2.json:rate2')
    custom_parser.add_argument(
        '--output',
        default='custom_plots.pdf',
        help=
        'Output PDF filename (default: custom_plots.pdf, saved in plots/ subdirectory)'
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "plot":
        plot_vllm_fig9_style(result_dir=args.dir,
                             ttft_slo=args.ttft_slo,
                             tpot_slo=args.tpot_slo,
                             atta_target=args.target)

    elif args.command == "custom":
        files_and_rates = []
        for file_rate in args.files:
            if ':' not in file_rate:
                print(
                    f"Invalid format: {file_rate}. Use filename:rate format.")
                sys.exit(1)
            filename, rate_str = file_rate.split(':', 1)
            rate = float(rate_str)
            files_and_rates.append((filename, rate))

        plot_custom(files_and_rates, args.ttft_slo, args.tpot_slo, args.output)


if __name__ == "__main__":
    main()
