#!/bin/bash

# vLLM LMCache Mode Benchmark and Plot Script
# 集成 LMCache 测试和可视化的一站式脚本

set -e

# Color definitions for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse command line arguments
DATASET_NAME=${1:-"sharegpt"}

if [[ "$DATASET_NAME" == "--help" || "$DATASET_NAME" == "-h" ]]; then
    echo "Usage: $0 [dataset_name]"
    echo ""
    echo "Arguments:"
    echo "  dataset_name    Dataset to use for benchmark (default: sharegpt)"
    echo ""
    echo "Environment Variables (LMCache configuration):"
    echo "  NUM_PREFILL_INSTANCES   Number of prefill instances (default: 1)"
    echo "  NUM_DECODE_INSTANCES    Number of decode instances (default: 1)"
    echo "  PREFILLER_TP_SIZE       Prefill tensor parallel size (default: 1)"
    echo "  DECODER_TP_SIZE         Decode tensor parallel size (default: 1)"
    echo "  LMCACHE_PORT           LMCache server port (default: 65432)"
    echo "  START_GPU_ID           Starting GPU ID (default: 0)"
    echo "  GPU_MEMORY_UTILIZATION GPU memory utilization (default: 0.8)"
    echo ""
    echo "Environment Variables (Benchmark configuration):"
    echo "  REQUEST_RATES          Custom request rates (default: 0.5,1.0,1.5,2.0,2.5,3.0)"
    echo "  TTFT_SLO              TTFT SLO threshold in ms (default: 125)"
    echo "  TPOT_SLO              TPOT SLO threshold in ms (default: 200)"
    echo "  TARGET_ATTAINMENT     Target attainment percentage (default: 90)"
    echo ""
    echo "Environment Variables (Output configuration):"
    echo "  PLOT_OUTPUT_DIR       Custom output directory for plots"
    echo ""
    echo "Examples:"
    echo "  $0 sharegpt"
    echo "  NUM_PREFILL_INSTANCES=2 NUM_DECODE_INSTANCES=2 $0 sharegpt"
    echo "  LMCACHE_PORT=65433 START_GPU_ID=4 $0 custom_dataset"
    exit 0
fi

print_info "Starting LMCache mode benchmark and plot workflow for dataset: $DATASET_NAME"

# Check if run_bench_lmcache.sh exists
if [[ ! -f "./run_bench_lmcache.sh" ]]; then
    print_error "run_bench_lmcache.sh not found in current directory"
    exit 1
fi

# Check if benchmark_plotter.py exists
if [[ ! -f "./benchmark_plotter.py" ]]; then
    print_error "benchmark_plotter.py not found in current directory"
    exit 1
fi

# Step 1: Run LMCache benchmark
print_info "Step 1: Running LMCache mode benchmark..."
print_info "This may take a while depending on the number of request rates and system performance"

if ./run_bench_lmcache.sh "$DATASET_NAME"; then
    print_success "LMCache benchmark completed successfully"
else
    print_error "LMCache benchmark failed"
    exit 1
fi

# Step 2: Find the latest result directory
print_info "Step 2: Locating latest LMCache benchmark results..."

RESULTS_BASE="./results/$DATASET_NAME"
if [[ ! -d "$RESULTS_BASE" ]]; then
    print_error "Results directory not found: $RESULTS_BASE"
    exit 1
fi

# Find the latest LMCache directory (should start with 'lmcache_' and be newest)
LATEST_DIR=$(find "$RESULTS_BASE" -maxdepth 1 -type d -name "lmcache_*" | sort | tail -1)

if [[ -z "$LATEST_DIR" ]]; then
    print_error "No LMCache result directories found in $RESULTS_BASE"
    print_info "Expected directories starting with 'lmcache_'"
    exit 1
fi

print_success "Found latest LMCache results: $LATEST_DIR"

# Step 3: Generate plots
print_info "Step 3: Generating performance plots..."

# Extract configuration for plot title
TTFT_SLO=${TTFT_SLO:-125}
TPOT_SLO=${TPOT_SLO:-200}
TARGET_ATTAINMENT=${TARGET_ATTAINMENT:-90}

print_info "Using SLO parameters: TTFT=${TTFT_SLO}ms, TPOT=${TPOT_SLO}ms, Target=${TARGET_ATTAINMENT}%"

if python3 benchmark_plotter.py \
    --result-dir "$LATEST_DIR" \
    --ttft-slo "$TTFT_SLO" \
    --tpot-slo "$TPOT_SLO" \
    --atta-target "$TARGET_ATTAINMENT"; then
    print_success "Performance plots generated successfully"
else
    print_error "Failed to generate performance plots"
    exit 1
fi

# Step 4: Show results summary
print_info "Step 4: Results Summary"
echo ""
print_success "================================================="
print_success "  LMCache Benchmark and Plot Workflow Complete"
print_success "================================================="
echo ""
print_info "📊 Results Location: $LATEST_DIR"
print_info "📈 Plots Location: $LATEST_DIR/plots/"

# Count result files
RESULT_COUNT=$(find "$LATEST_DIR" -name "vllm-*.json" | wc -l)
print_info "📋 Test Results: $RESULT_COUNT JSON files"

# Show plots if they exist
PLOTS_DIR="$LATEST_DIR/plots"
if [[ -d "$PLOTS_DIR" ]]; then
    PLOT_COUNT=$(find "$PLOTS_DIR" -name "*.pdf" | wc -l)
    print_info "🎯 Generated Plots: $PLOT_COUNT PDF files"
    
    if [[ $PLOT_COUNT -gt 0 ]]; then
        echo ""
        print_info "📈 Available Plots:"
        find "$PLOTS_DIR" -name "*.pdf" | sort | while read -r plot_file; do
            basename_plot=$(basename "$plot_file")
            print_info "   - $basename_plot"
        done
    fi
else
    print_warning "Plots directory not found: $PLOTS_DIR"
fi

echo ""
print_success "🚀 To compare with other modes, use:"
print_info "python3 benchmark_plotter.py --compare \\"
print_info "    --lmcache-dir $LATEST_DIR \\"
print_info "    --simple-dir /path/to/simple/results \\"
print_info "    --nixl-dir /path/to/nixl/results"

echo ""
print_success "✨ LMCache benchmark workflow completed successfully!"
