# Performance Analysis

这个文件夹包含了用于分析 vLLM 分布式服务性能的脚本和工具。主要用于比较不同配置下的延迟、吞吐量和 SLO 达成情况。

## 文件夹结构

```
performance_analysis/
├── scripts/           # 性能测试脚本
│   ├── simple/       # 简单配置测试脚本
│   ├── p2p_nccl/     # P2P NCCL 配置测试脚本
│   ├── nixl/         # NIXL 配置测试脚本
│   └── generate_plots.sh  # 生成所有图表的主脚本
├── ploters/          # 数据可视化脚本
└── report.md         # 性能分析报告
```

## 主要功能

### 1. 性能测试脚本 (`scripts/`)

每个配置文件夹包含以下类型的脚本：

- **`serve.sh`** - 启动 vLLM 服务器
- **`run.sh`** - 执行完整的性能测试流程
- **`*_bench.sh`** - 针对特定数据集的基准测试
  - `random-512-64.sh` - 随机数据测试
  - `sharegpt.sh` - ShareGPT 数据集测试
  - `gsm8k.sh` - GSM8K 数据集测试
  - `human_eval.sh` - HumanEval 数据集测试

### 2. 数据可视化 (`ploters/`)

- **`plot_latency_breakdown.py`** - 延迟分解分析图
- **`plot_latency_rps_per_gpu_comparison.py`** - 每 GPU RPS 延迟对比图
- **`plot_slo_attainment_rps_per_gpu_comparison.py`** - SLO 达成率对比图

### 3. 配置类型

- **Simple** - 标准 vLLM 服务配置
- **P2P NCCL** - 使用 P2P NCCL 的分布式配置
- **NIXL** - NIXL 网络配置

## 快速开始

### 1. 运行性能测试

```bash
# 运行简单配置测试
cd scripts/simple
./run.sh

# 运行 P2P NCCL 配置测试
cd scripts/p2p_nccl
./run.sh

# 运行 NIXL 配置测试
cd scripts/nixl
./run.sh
```

### 2. 生成性能分析图表

```bash
# 生成所有配置的对比图表
cd scripts
./generate_plots.sh
```

生成的图表将保存在 `../plots/` 目录下，按数据集分类。

### 3. 自定义配置

可以通过环境变量自定义测试参数：

```bash
export MODEL_PATH=/path/to/your/model
export GPU_ID=0
export SERVER_PORT=8027
export REQUEST_RATES="1.0 2.0 3.0 4.0 5.0"
```

## 输出结果

- **JSON 结果文件** - 详细的性能指标数据
- **PNG 图表** - 可视化的性能对比图
- **延迟分析** - TTFT (Time To First Token) 和 TPOT (Time Per Output Token) 分解
- **SLO 分析** - 服务级别目标达成情况

## 注意事项

- 确保在运行测试前已正确配置模型路径
- 多 GPU 配置需要相应的硬件支持
- 测试结果受硬件配置和网络环境影响
- **KV 传输时间统计限制**：目前 KV cache 传输时间统计功能仅支持 1P1D（1个 prefill instance + 1个 decode instance）配置。在其他分布式配置下，KV 传输时间可能无法正确统计或显示
