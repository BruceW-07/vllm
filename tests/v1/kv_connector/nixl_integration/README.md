# vLLM Benchmark Tools

🚀 **高效的 vLLM 性能评估和可视化工具套件**

支持标准单实例模式（Simple）和前缀-解码分离模式（NIXL），提供自动化测试、结果可视化和性能对比功能。

## ⚡ 快速开始

### 🔄 NIXL模式（前缀-解码分离）
```bash
# 基本测试
./run_benchmark_and_plot_nixl.sh sharegpt

# 自定义配置
NUM_PREFILL_INSTANCES=2 NUM_DECODE_INSTANCES=2 \
./run_benchmark_and_plot_nixl.sh sharegpt
```

### 🔧 Simple模式（标准单实例）
```bash
# 基本测试
./run_benchmark_and_plot_simple.sh sharegpt

# 自定义配置
TENSOR_PARALLEL_SIZE=4 \
./run_benchmark_and_plot_simple.sh sharegpt
```

## 📊 性能对比（核心功能）
```bash
# 对比两种模式的性能差异
python benchmark_plotter.py --compare \
    --simple-dir ./results/sharegpt/simple_20240814_143022 \
    --nixl-dir ./results/sharegpt/nixl_20240814_150000

# 自定义SLO参数对比
python benchmark_plotter.py --compare \
    --simple-dir ./results/simple_results \
    --nixl-dir ./results/nixl_results \
    --ttft-slo 100 --tpot-slo 150 --atta-target 95
```

## 🎯 主要特性

- ✅ **双模式支持**：Simple（标准）和 NIXL（前缀-解码分离）
- ✅ **自动化测试**：一键运行benchmark + 可视化
- ✅ **性能对比**：直观对比两种部署模式的性能差异
- ✅ **结果管理**：时间戳目录结构，避免文件冲突
- ✅ **配置记录**：自动记录和展示关键配置信息
- ✅ **服务器友好**：无GUI依赖，适合远程服务器环境

## � 脚本说明

| 脚本 | 功能 | 使用场景 |
|------|------|----------|
| `run_benchmark_and_plot_nixl.sh` | NIXL模式自动化测试 | 高性能分离测试 |
| `run_benchmark_and_plot_simple.sh` | Simple模式自动化测试 | 标准基线测试 |
| `benchmark_plotter.py` | 结果可视化和对比工具 | 单一模式分析或双模式对比 |
| `list_results.sh` | 查看历史测试结果 | 结果管理 |

## ⚙️ 配置参数

### NIXL模式参数
| 参数 | 默认值 | 描述 |
|------|-------|------|
| `NUM_PREFILL_INSTANCES` | 1 | prefill实例数量 |
| `NUM_DECODE_INSTANCES` | 1 | decode实例数量 |
| `PREFILLER_TP_SIZE` | 1 | prefill张量并行大小 |
| `DECODER_TP_SIZE` | 1 | decode张量并行大小 |

### Simple模式参数
| 参数 | 默认值 | 描述 |
|------|-------|------|
| `TENSOR_PARALLEL_SIZE` | 1 | 张量并行大小 |
| `GPU_MEMORY_UTILIZATION` | 0.8 | GPU内存利用率 |

### SLO参数（通用）
| 参数 | 默认值 | 描述 |
|------|-------|------|
| `TTFT_SLO` | 125 | TTFT SLO阈值（毫秒） |
| `TPOT_SLO` | 200 | TPOT SLO阈值（毫秒） |
| `TARGET_ATTAINMENT` | 90 | 目标达成率（%） |

## � 可视化功能

### 单一模式图表
生成类似 DistServe Figure 9 的性能图表：
- **左子图**：SLO达成率 vs 请求率
- **右子图**：SLO达成率 vs SLO Scale
- **三条线**：Both TTFT & TPOT、TTFT only、TPOT only

### 对比模式图表（新功能）
在同一张图中对比两种模式的性能：
- **左子图**：两种模式的请求率对比
- **右子图**：两种模式的SLO Scale对比
- **颜色区分**：Simple模式（红色），NIXL模式（蓝色）
- **交点标注**：自动标注性能交点

## �📁 结果文件结构

```
results/
├── sharegpt/
│   ├── simple_20240814_143022/     # Simple模式结果
│   │   ├── vllm-0.5qps-*.json      # 各QPS测试结果
│   │   ├── vllm-1.0qps-*.json
│   │   └── plots/
│   │       └── vllm_benchmark_plots.pdf
│   └── nixl_20240814_150000/       # NIXL模式结果
│       ├── vllm-0.5qps-*.json
│       └── plots/
│           └── vllm_comparison_plots.pdf
```

### JSON结果文件格式
```json
{
  "ttfts": [0.123, 0.145, ...],      // 首字符时间（秒）
  "tpots": [0.089, 0.091, ...],      // 输出令牌时间（秒）
  "model_name": "Qwen3-0.6B",        // 模型名称
  "dataset_name": "sharegpt",        // 数据集名称
  "deployment_mode": "simple/nixl",  // 部署模式
  "num_prefill_instances": 1,        // 配置信息
  "tensor_parallel_size": 1
}
```

## �️ 详细使用方法

### 1. 单一模式测试

#### NIXL模式（高性能分离）
```bash
# 基本测试
./run_benchmark_and_plot_nixl.sh sharegpt

# 自定义实例配置
NUM_PREFILL_INSTANCES=2 NUM_DECODE_INSTANCES=3 \
PREFILLER_TP_SIZE=2 DECODER_TP_SIZE=1 \
./run_benchmark_and_plot_nixl.sh sharegpt

# 完整自定义
NUM_PREFILL_INSTANCES=2 NUM_DECODE_INSTANCES=2 \
PREFILLER_TP_SIZE=2 DECODER_TP_SIZE=2 \
TTFT_SLO=100 TPOT_SLO=150 TARGET_ATTAINMENT=95 \
./run_benchmark_and_plot_nixl.sh my_dataset
```

#### Simple模式（标准基线）
```bash
# 基本测试
./run_benchmark_and_plot_simple.sh sharegpt

# 自定义配置
TENSOR_PARALLEL_SIZE=4 GPU_MEMORY_UTILIZATION=0.9 \
TTFT_SLO=100 TPOT_SLO=150 TARGET_ATTAINMENT=95 \
./run_benchmark_and_plot_simple.sh my_dataset
```

### 2. 性能对比分析

#### 使用 benchmark_plotter.py
```bash
# 单一模式结果可视化
python benchmark_plotter.py --result-dir ./results/sharegpt/20240814_143022

# 两种模式性能对比
python benchmark_plotter.py --compare \
    --simple-dir ./results/sharegpt/simple_20240814_143022 \
    --nixl-dir ./results/sharegpt/nixl_20240814_150000

# 查看所有参数选项
python benchmark_plotter.py --help
```

### 3. 结果管理
```bash
# 查看历史测试结果
./list_results.sh

# 输出示例：
# 2024-08-14 15:30:22 - sharegpt/nixl_20240814_153022 (5 files, 1 plots)
#   Config: PF:2, D:2, TP:2x2, Model:Qwen3-0.6B
# 2024-08-14 14:30:22 - sharegpt/simple_20240814_143022 (5 files, 1 plots)  
#   Config: TP:1, Model:Qwen3-0.6B
```

## 🎯 算法实现

### SLO达成率计算
```python
def get_attainment(ttfts, tpots, ttft_slo, tpot_slo) -> float:
    """计算同时满足TTFT和TPOT SLO的请求比例"""
    counter = 0
    for ttft, tpot in zip(ttfts, tpots):
        if ttft <= ttft_slo and tpot <= tpot_slo:
            counter += 1
    return (counter / len(ttfts)) * 100
```

### 交点检测
```python
def find_intersection(xs, ys, target_y):
    """线性插值找到曲线与目标线的交点"""
    for i in range(len(xs) - 1):
        if (ys[i] < target_y) != (ys[i+1] < target_y):
            inter_x = (target_y - ys[i]) * (xs[i+1] - xs[i]) / (ys[i+1] - ys[i]) + xs[i]
            return (inter_x, target_y)
    return (None, None)
```

## 💡 最佳实践

### 性能评估流程
1. **基线测试**：先运行Simple模式建立性能基线
2. **高级测试**：运行NIXL模式评估前缀-解码分离的收益
3. **结果对比**：使用对比功能直观评估两种模式的性能差异
4. **参数调优**：根据结果调整实例数量和SLO阈值

### 资源规划
- **NIXL模式**：需要更多GPU资源，适合高吞吐量场景
- **Simple模式**：资源需求较低，适合快速验证和基线测试
- **对比分析**：结合两种模式的结果进行全面性能评估

### 注意事项
- 确保GPU资源充足，特别是NIXL模式的多实例配置
- 使用相同的数据集和SLO参数进行公平对比
- 定期清理历史结果文件，避免磁盘空间不足
- 在服务器环境中设置合适的 `PLOT_OUTPUT_DIR` 环境变量

## 🔧 环境依赖

- Python 3.8+
- matplotlib, numpy（用于可视化）
- vLLM框架
- CUDA支持的GPU
