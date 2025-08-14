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
                color="#5470c6",
                marker="o",
                linewidth=2,
                markersize=6)
        ax.plot(xs,
                ys_ttft,
                label="TTFT only",
                linestyle="-",
                color="#ee6666",
                marker="s",
                linewidth=2,
                markersize=6)
        ax.plot(xs,
                ys_tpot,
                label="TPOT only",
                linestyle="-",
                color="#4dc832",
                marker="^",
                linewidth=2,
                markersize=6)

        if atta_target:
            # Draw target line
            ax.axhline(y=atta_target, color="grey", linestyle="--", alpha=0.7)
            
            # Check and draw intersection lines for each curve
            try:
                inter_x, inter_y = find_intersection(xs, ys_both, atta_target)
                if inter_x is not None:
                    ax.vlines(x=inter_x, ymin=0, ymax=inter_y,
                             linestyles="--", colors="#5470c6", alpha=0.8)
            except:
                pass
                
            try:
                inter_x_ttft, inter_y_ttft = find_intersection(xs, ys_ttft, atta_target)
                if inter_x_ttft is not None:
                    ax.vlines(x=inter_x_ttft, ymin=0, ymax=inter_y_ttft,
                             linestyles="--", colors="#ee6666", alpha=0.8)
            except:
                pass
                
            try:
                inter_x_tpot, inter_y_tpot = find_intersection(xs, ys_tpot, atta_target)
                if inter_x_tpot is not None:
                    ax.vlines(x=inter_x_tpot, ymin=0, ymax=inter_y_tpot,
                             linestyles="--", colors="#4dc832", alpha=0.8)
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
                color="#5470c6",
                marker="o",
                linewidth=2,
                markersize=6)
        ax.plot(xs,
                ys_ttft,
                label="TTFT only",
                linestyle="-",
                color="#ee6666",
                marker="s",
                linewidth=2,
                markersize=6)
        ax.plot(xs,
                ys_tpot,
                label="TPOT only",
                linestyle="-",
                color="#4dc832",
                marker="^",
                linewidth=2,
                markersize=6)

        if atta_target:
            # Draw target line
            ax.axhline(y=atta_target, color="grey", linestyle="--", alpha=0.7)
            
            # Check and draw intersection lines for each curve
            try:
                inter_x, inter_y = find_intersection(xs, ys_both, atta_target)
                if inter_x is not None:
                    ax.vlines(x=inter_x, ymin=0, ymax=inter_y,
                             linestyles="--", colors="#5470c6", alpha=0.8)
            except:
                pass
                
            try:
                inter_x_ttft, inter_y_ttft = find_intersection(xs, ys_ttft, atta_target)
                if inter_x_ttft is not None:
                    ax.vlines(x=inter_x_ttft, ymin=0, ymax=inter_y_ttft,
                             linestyles="--", colors="#ee6666", alpha=0.8)
            except:
                pass
                
            try:
                inter_x_tpot, inter_y_tpot = find_intersection(xs, ys_tpot, atta_target)
                if inter_x_tpot is not None:
                    ax.vlines(x=inter_x_tpot, ymin=0, ymax=inter_y_tpot,
                             linestyles="--", colors="#4dc832", alpha=0.8)
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
            if 'model_name' in data:
                title_parts.append(f"Model:{data['model_name']}")
            elif 'model' in data:
                # Fallback to extracting from model path
                model_path = data['model']
                model_name = model_path.split('/')[-1] if '/' in model_path else model_path
                title_parts.append(f"Model:{model_name}")
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
        axs[0].set_title(f"SLO Attainment vs Request Rate\n(TTFT≤{ttft_slo}ms, TPOT≤{tpot_slo}ms)", 
                         fontsize=14, pad=20)

        # Second subplot: SLO scale plot using middle file
        middle_idx = len(result_files) // 2
        request_rate = request_rates[0]
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
        axs[1].set_title(f"SLO Attainment vs SLO Scale (Request Rate: {request_rate:.1f} req/s)\n(TTFT≤{ttft_slo}ms, TPOT≤{tpot_slo}ms)", 
                         fontsize=14, pad=20)

        # Add overall title (closer to subplots)
        fig.suptitle(title_info, fontsize=18, y=0.97)

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
                   bbox_to_anchor=(0.5, 0.08),
                   ncol=3, 
                   frameon=False,
                   fontsize=14)
    else:
        print(f"No result files found in {result_dir}")
        return

    # Adjust layout to accommodate title and legend with more space for plots
    plt.subplots_adjust(top=0.90, bottom=0.20)
    plt.tight_layout(rect=[0, 0.17, 1, 0.92])

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

    # Generate filename from result directory name
    result_dirname = os.path.basename(result_dir.rstrip('/'))
    output_filename = f"{result_dirname}_benchmark_plots.pdf"
    
    output_path = os.path.join(plots_dir, output_filename)
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    # Don't show plot in headless environment
    # plt.show()


def plot_comparison(simple_dir: str,
                   nixl_dir: str,
                   ttft_slo: float = 125.0,
                   tpot_slo: float = 200.0,
                   atta_target: float = 90.0,
                   use_separate_rates: bool = False):
    """Create comparison plots between Simple mode and NIXL mode
    
    Args:
        simple_dir: Directory containing Simple mode result files
        nixl_dir: Directory containing NIXL mode result files
        ttft_slo: TTFT SLO threshold in milliseconds (default: 125ms)
        tpot_slo: TPOT SLO threshold in milliseconds (default: 200ms)
        atta_target: Target attainment rate percentage (default: 90%)
        use_separate_rates: Use separate request rate sets instead of intersection (default: False)
    """
    if not HAS_PLOTTING:
        print("Cannot create plots - matplotlib/numpy not available")
        return

    plt.rcParams.update({'font.size': 16})
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    # Scan directories for result files
    simple_files_and_rates = scan_result_files(simple_dir)
    nixl_files_and_rates = scan_result_files(nixl_dir)

    if not simple_files_and_rates:
        print(f"No result files found in Simple directory: {simple_dir}")
        return
    
    if not nixl_files_and_rates:
        print(f"No result files found in NIXL directory: {nixl_dir}")
        return

    print(f"Found {len(simple_files_and_rates)} Simple mode result files")
    print(f"Found {len(nixl_files_and_rates)} NIXL mode result files")

    # Build title with configuration info from both modes
    title_info = ""
    try:
        simple_data = load_vllm_result(simple_files_and_rates[0][0])
        nixl_data = load_vllm_result(nixl_files_and_rates[0][0])
        
        # Extract model name
        model_name = ""
        if 'model_name' in simple_data:
            model_name = simple_data['model_name']
        elif 'model' in simple_data:
            model_name = simple_data['model'].split('/')[-1]
        
        # Extract dataset name
        dataset_name = simple_data.get('dataset_name', 'unknown')
        
        title_info = f"Performance Comparison: Simple vs NIXL Mode (Model:{model_name}, Dataset:{dataset_name})"
                
    except Exception as e:
        title_info = "Performance Comparison: Simple vs NIXL Mode"
        print(f"Could not extract config for title: {e}")

    # Prepare data for plotting with aligned request rates
    simple_dict = {r: f for f, r in simple_files_and_rates}
    nixl_dict = {r: f for f, r in nixl_files_and_rates}
    
    # Find common request rates for fair comparison
    simple_rates_set = set(simple_dict.keys())
    nixl_rates_set = set(nixl_dict.keys())
    common_rates = sorted(simple_rates_set.intersection(nixl_rates_set))
    
    print(f"Simple rates: {sorted(simple_rates_set)}")
    print(f"NIXL rates: {sorted(nixl_rates_set)}")
    print(f"Common rates for comparison: {common_rates}")
    
    if use_separate_rates or not common_rates:
        if use_separate_rates:
            print("Using separate request rate sets as requested")
        else:
            print("Warning: No common request rates found between Simple and NIXL results")
            print("Will plot with separate rate sets, but comparison may not be meaningful")
        # Use original behavior with separate rate sets
        simple_result_files = [f for f, r in simple_files_and_rates]
        simple_request_rates = [r for f, r in simple_files_and_rates]
        nixl_result_files = [f for f, r in nixl_files_and_rates]
        nixl_request_rates = [r for f, r in nixl_files_and_rates]
    else:
        # Use only common rates for fair comparison (default behavior)
        print(f"Using common request rates for fair comparison: {len(common_rates)} rates")
        simple_result_files = [simple_dict[r] for r in common_rates]
        simple_request_rates = common_rates
        nixl_result_files = [nixl_dict[r] for r in common_rates]
        nixl_request_rates = common_rates

    # Define backends for comparison
    simple_backend = Backend("simple", "Simple Mode", "#ee6666")
    nixl_backend = Backend("nixl", "NIXL Mode", "#5470c6")

    # First subplot: Attainment rate comparison
    draw_comparison_attainment_plot(axs[0],
                                  simple_result_files, simple_request_rates, simple_backend,
                                  nixl_result_files, nixl_request_rates, nixl_backend,
                                  ttft_slo=ttft_slo,
                                  tpot_slo=tpot_slo,
                                  atta_target=atta_target,
                                  show_ylabel=True)
    axs[0].set_title(f"SLO Attainment vs Request Rate\n(TTFT≤{ttft_slo}ms, TPOT≤{tpot_slo}ms)", 
                     fontsize=14, pad=20)

    # Second subplot: SLO scale comparison using middle files
    simple_middle_idx = len(simple_result_files) // 2
    nixl_middle_idx = len(nixl_result_files) // 2
    
    draw_comparison_slo_scale_plot(axs[1],
                                 simple_result_files[simple_middle_idx], simple_backend,
                                 nixl_result_files[nixl_middle_idx], nixl_backend,
                                 ttft_slo=ttft_slo,
                                 tpot_slo=tpot_slo,
                                 scales=[1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
                                 atta_target=atta_target,
                                 show_ylabel=True)
    
    simple_middle_rate = simple_request_rates[simple_middle_idx]
    nixl_middle_rate = nixl_request_rates[nixl_middle_idx]
    axs[1].set_title(f"SLO Attainment vs SLO Scale\n(Request Rates: Simple={simple_middle_rate:.1f}, NIXL={nixl_middle_rate:.1f} req/s)", 
                     fontsize=14, pad=20)

    # Add overall title
    fig.suptitle(title_info, fontsize=18, y=0.97)

    # Add legend for comparison
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, 
               loc='lower center', 
               bbox_to_anchor=(0.5, 0.08),
               ncol=2, 
               frameon=False,
               fontsize=14)

    # Adjust layout
    plt.subplots_adjust(top=0.90, bottom=0.20)
    plt.tight_layout(rect=[0, 0.17, 1, 0.92])

    # Save plot
    custom_output_dir = os.environ.get('PLOT_OUTPUT_DIR')
    
    if custom_output_dir:
        plots_dir = custom_output_dir
        os.makedirs(plots_dir, exist_ok=True)
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        plots_dir = os.path.join(script_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

    # Generate filename from combined directory names
    simple_dirname = os.path.basename(simple_dir.rstrip('/'))
    nixl_dirname = os.path.basename(nixl_dir.rstrip('/'))
    combined_filename = f"{simple_dirname}_vs_{nixl_dirname}_comparison.pdf"
    
    output_path = os.path.join(plots_dir, combined_filename)
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Comparison plot saved to {output_path}")


def draw_comparison_attainment_plot(ax,
                                  simple_files: List[str], simple_rates: List[float], simple_backend: Backend,
                                  nixl_files: List[str], nixl_rates: List[float], nixl_backend: Backend,
                                  ttft_slo: float, tpot_slo: float,
                                  atta_target: Optional[float] = None,
                                  show_ylabel: bool = False):
    """Draw attainment rate comparison plot between Simple and NIXL modes"""
    if not HAS_PLOTTING:
        return

    ax.set_xlabel("Request Rate (req/s)")
    if show_ylabel:
        ax.set_ylabel("SLO Attainment (%)")

    # Plot Simple mode results
    simple_ys_both = []
    for result_file in simple_files:
        try:
            data = load_vllm_result(result_file)
            ttfts = [t * 1000 for t in data.get('ttfts', [])]
            tpots = [t * 1000 for t in data.get('tpots', [])]
            
            if ttfts and tpots:
                simple_ys_both.append(get_attainment(ttfts, tpots, ttft_slo, tpot_slo))
            else:
                simple_ys_both.append(0)
        except Exception as e:
            print(f"Error processing Simple file {result_file}: {e}")
            simple_ys_both.append(0)

    # Plot NIXL mode results
    nixl_ys_both = []
    for result_file in nixl_files:
        try:
            data = load_vllm_result(result_file)
            ttfts = [t * 1000 for t in data.get('ttfts', [])]
            tpots = [t * 1000 for t in data.get('tpots', [])]
            
            if ttfts and tpots:
                nixl_ys_both.append(get_attainment(ttfts, tpots, ttft_slo, tpot_slo))
            else:
                nixl_ys_both.append(0)
        except Exception as e:
            print(f"Error processing NIXL file {result_file}: {e}")
            nixl_ys_both.append(0)

    # Plot lines
    ax.plot(simple_rates, simple_ys_both,
            label=simple_backend.label,
            color=simple_backend.color,
            marker="o",
            linewidth=2,
            markersize=6)
    
    ax.plot(nixl_rates, nixl_ys_both,
            label=nixl_backend.label,
            color=nixl_backend.color,
            marker="s",
            linewidth=2,
            markersize=6)

    # Add target line and intersections if specified
    if atta_target:
        ax.axhline(y=atta_target, color="grey", linestyle="--", alpha=0.7)
        
        # Draw intersection lines
        try:
            inter_x, inter_y = find_intersection(simple_rates, simple_ys_both, atta_target)
            if inter_x is not None:
                ax.vlines(x=inter_x, ymin=0, ymax=inter_y,
                         linestyles="--", colors=simple_backend.color, alpha=0.8)
        except:
            pass
            
        try:
            inter_x, inter_y = find_intersection(nixl_rates, nixl_ys_both, atta_target)
            if inter_x is not None:
                ax.vlines(x=inter_x, ymin=0, ymax=inter_y,
                         linestyles="--", colors=nixl_backend.color, alpha=0.8)
        except:
            pass

    ax.set_ylim(0, 105)


def draw_comparison_slo_scale_plot(ax,
                                 simple_file: str, simple_backend: Backend,
                                 nixl_file: str, nixl_backend: Backend,
                                 ttft_slo: float, tpot_slo: float,
                                 scales: List[float],
                                 atta_target: Optional[float] = None,
                                 show_ylabel: bool = False):
    """Draw SLO scale comparison plot between Simple and NIXL modes"""
    if not HAS_PLOTTING:
        return

    ax.set_xlabel("SLO Scale")
    if show_ylabel:
        ax.set_ylabel("SLO Attainment (%)")

    scales = sorted(scales, reverse=True)
    ax.invert_xaxis()
    ax.set_ylim(0, 105)

    # Process Simple mode data
    try:
        simple_data = load_vllm_result(simple_file)
        simple_ttfts = [t * 1000 for t in simple_data.get('ttfts', [])]
        simple_tpots = [t * 1000 for t in simple_data.get('tpots', [])]
        
        if simple_ttfts and simple_tpots:
            simple_ys = []
            for scale in scales:
                simple_ys.append(get_attainment(simple_ttfts, simple_tpots, 
                                              ttft_slo * scale, tpot_slo * scale))
            
            ax.plot(scales, simple_ys,
                    label=simple_backend.label,
                    color=simple_backend.color,
                    marker="o",
                    linewidth=2,
                    markersize=6)
    except Exception as e:
        print(f"Error processing Simple file for SLO scale: {e}")

    # Process NIXL mode data
    try:
        nixl_data = load_vllm_result(nixl_file)
        nixl_ttfts = [t * 1000 for t in nixl_data.get('ttfts', [])]
        nixl_tpots = [t * 1000 for t in nixl_data.get('tpots', [])]
        
        if nixl_ttfts and nixl_tpots:
            nixl_ys = []
            for scale in scales:
                nixl_ys.append(get_attainment(nixl_ttfts, nixl_tpots, 
                                            ttft_slo * scale, tpot_slo * scale))
            
            ax.plot(scales, nixl_ys,
                    label=nixl_backend.label,
                    color=nixl_backend.color,
                    marker="s",
                    linewidth=2,
                    markersize=6)
    except Exception as e:
        print(f"Error processing NIXL file for SLO scale: {e}")

    # Add target line if specified
    if atta_target:
        ax.axhline(y=atta_target, color="grey", linestyle="--", alpha=0.7)


def main():
    """Main function to parse command line arguments and generate plots"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot vLLM benchmark results")
    parser.add_argument("--result-dir", type=str, 
                       help="Directory containing benchmark result files")
    parser.add_argument("--ttft-slo", type=float, default=125.0,
                       help="TTFT SLO threshold in milliseconds (default: 125.0)")
    parser.add_argument("--tpot-slo", type=float, default=200.0,
                       help="TPOT SLO threshold in milliseconds (default: 200.0)")
    parser.add_argument("--atta-target", type=float, default=90.0,
                       help="Target attainment percentage (default: 90.0)")
    
    # Comparison mode arguments
    parser.add_argument("--compare", action="store_true",
                       help="Enable comparison mode between Simple and NIXL results")
    parser.add_argument("--simple-dir", type=str,
                       help="Directory containing Simple mode benchmark results")
    parser.add_argument("--nixl-dir", type=str,
                       help="Directory containing NIXL mode benchmark results")
    parser.add_argument("--use-separate-rates", action="store_true",
                       help="Use separate request rate sets instead of common intersection (default: use intersection)")
    
    args = parser.parse_args()
    
    if args.compare:
        # Comparison mode
        if not args.simple_dir or not args.nixl_dir:
            print("Error: --simple-dir and --nixl-dir are required for comparison mode")
            return 1
        
        plot_comparison(simple_dir=args.simple_dir,
                       nixl_dir=args.nixl_dir,
                       ttft_slo=args.ttft_slo,
                       tpot_slo=args.tpot_slo,
                       atta_target=args.atta_target,
                       use_separate_rates=args.use_separate_rates)
    else:
        # Normal mode
        if not args.result_dir:
            print("Error: --result-dir is required for normal mode")
            return 1
        
        plot_vllm_fig9_style(result_dir=args.result_dir,
                             ttft_slo=args.ttft_slo,
                             tpot_slo=args.tpot_slo,
                             atta_target=args.atta_target)
    
    return 0


if __name__ == "__main__":
    exit(main())
