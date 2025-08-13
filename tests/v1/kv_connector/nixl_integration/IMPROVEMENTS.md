## vLLM Benchmark 脚本改进总结

### 主要改进

#### 1. 时间戳目录结构
- **结果文件按时间组织**：每次运行都创建新的时间戳目录 (`results/{dataset}/{timestamp}/`)
- **避免文件冲突**：多次测试不会互相覆盖
- **图片也保存在对应时间文件夹**：`results/{dataset}/{timestamp}/plots/`

#### 2. 配置参数管理
- **Shell脚本参数**：在 `run_benchmark_and_plot.sh` 中集中设置绘图参数
  - `TTFT_SLO`：TTFT SLO阈值（默认125ms）
  - `TPOT_SLO`：TPOT SLO阈值（默认200ms）
  - `TARGET_ATTAINMENT`：目标达成率（默认90%）

#### 3. 图表改进
- **简化图例**：只显示"Both TTFT & TPOT"、"TTFT only"、"TPOT only"三条线的区别
- **配置信息在标题中**：PF、D、TP等配置信息显示在图表总标题中
- **居中布局**：图例位于底部居中，标题在顶部，布局更美观
- **无弹窗显示**：适合服务器环境，只保存PDF文件

#### 4. 元数据记录
- **完整配置记录**：在JSON结果文件中记录所有关键配置参数
- **文件名包含配置**：便于快速识别不同配置的测试结果

### 使用示例

```bash
# 使用默认参数
./run_benchmark_and_plot.sh

# 自定义实例配置
NUM_PREFILL_INSTANCES=2 NUM_DECODE_INSTANCES=2 ./run_benchmark_and_plot.sh

# 自定义绘图参数
TTFT_SLO=100 TPOT_SLO=150 TARGET_ATTAINMENT=95 ./run_benchmark_and_plot.sh my_dataset

# 完整自定义
NUM_PREFILL_INSTANCES=2 NUM_DECODE_INSTANCES=2 \
PREFILLER_TP_SIZE=2 DECODER_TP_SIZE=2 \
TTFT_SLO=100 TPOT_SLO=150 TARGET_ATTAINMENT=95 \
./run_benchmark_and_plot.sh my_dataset
```

### 目录结构

```
results/
├── sharegpt/
│   ├── 20250813_142530/
│   │   ├── vllm-0.5qps-*.json
│   │   ├── vllm-1.0qps-*.json
│   │   ├── ...
│   │   └── plots/
│   │       └── vllm_benchmark_plots.pdf
│   └── 20250813_153045/
│       ├── vllm-0.5qps-*.json
│       └── plots/
│           └── vllm_benchmark_plots.pdf
└── my_dataset/
    └── 20250813_164512/
        ├── ...
        └── plots/
```

### 图表特性

1. **总标题**：显示配置信息（如："vLLM Benchmark Results (PF:2, D:2, TP:2x2, Dataset:sharegpt)"）
2. **子图标题**：
   - 左图："SLO Attainment vs Request Rate"
   - 右图："SLO Attainment vs SLO Scale"
3. **简化图例**：
   - "Both TTFT & TPOT"（实线）
   - "TTFT only"（点线）
   - "TPOT only"（虚线）
4. **居中布局**：图例在底部居中，整体布局更清晰

### 环境变量支持

- `PLOT_OUTPUT_DIR`：自定义图片输出目录
- 实例配置：`NUM_PREFILL_INSTANCES`, `NUM_DECODE_INSTANCES`, `PREFILLER_TP_SIZE`, `DECODER_TP_SIZE`
- 绘图参数：`TTFT_SLO`, `TPOT_SLO`, `TARGET_ATTAINMENT`

### 兼容性

- 保持与原有脚本的向后兼容性
- 支持headless环境（服务器环境）
- 自动处理依赖缺失情况
