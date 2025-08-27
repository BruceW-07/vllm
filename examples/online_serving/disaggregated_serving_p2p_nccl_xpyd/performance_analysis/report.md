# vLLM PD分离服务性能测试报告

## 1. 简介

本报告记录了使用 P2P NCCL 进行 KV 传输的 vLLM PD分离服务实现的性能测试。测试比较了PD合并服务配置与使用 P2P NCCL KV 传输的PD分离服务配置之间的性能。

## 2. 实验设置

### 2.1 硬件配置
- GPU: A100-80G

### 2.2 软件配置
- vLLM 版本: 未指定
- Python 库: vllm, pandas, datasets, quart
- 模型: Llama-3.1-8B-Instruct (位于 `/workspace/models/Llama-3.1-8B-Instruct`)

### 2.3 测试参数

#### 2.3.1 PD合并服务配置
- tensor-parallel-size: 1
- gpu-memory-utilization: 0.9
- max-model-len: 10000
- max-num-batched-tokens: 10000
- max-num-seqs: 256
- dtype: float16
- seed: 1024

#### 2.3.2 PD分离服务配置 (P2P NCCL)
- Architecture: 1 Prefill + 3 Decode (1P3D)
- Prefill GPUs: 0
- Decode GPUs: 1
- Prefill ports: 20003
- Decode ports: 20005
- Proxy service discovery port: 30001
- Proxy app port: 10001
- KV transfer config:
  - kv_connector: P2pNcclConnector
  - kv_role: kv_producer/kv_consumer
  - kv_buffer_size: 1e1 (producer), 8e9 (consumer)
  - nccl_num_channels: 16

#### 2.3.3 基准测试参数
- dataset-name: random
- random-input-len: 512 tokens
- random-output-len: 64 tokens
- ignore-eos: enabled
- metric-percentiles: 90, 95, 99
- seed: 1024
- Request rates:
  - PD-Combined serving: 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5
  - PD-Separated serving: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
- Duration: 300 seconds per request rate (calculated as request_rate * 300)

## 3. 性能结果

### 3.1 延迟分解分析

#### 3.1.1 PD合并服务配置
![延迟分解 - PD合并](plots/latency_breakdown.png)

#### 3.1.2 PD分离服务配置 (P2P NCCL)
![延迟分解 - PD分离](plots/latency_breakdown.png)

### 3.2 TTFT 分解分析

#### 3.2.1 PD合并服务配置
![TTFT 分解 - PD合并](plots/ttft_breakdown.png)

#### 3.2.2 PD分离服务配置 (P2P NCCL)
![TTFT 分解 - PD分离](plots/ttft_breakdown.png)

### 3.3 延迟比较 (每 GPU 的请求速率)

![延迟比较 - P90](plots/latency_rps_per_gpu_comparison_p90.png)
![延迟比较 - P95](plots/latency_rps_per_gpu_comparison_p95.png)
![延迟比较 - P99](plots/latency_rps_per_gpu_comparison_p99.png)

### 3.4 SLO 达成率比较

![SLO 达成率 - TTFT 和 TPOT](plots/slo_attainment_rps_per_gpu_comparison.png)

## 4. 详细指标

### 4.1 PD合并服务配置
| 请求速率 (req/s) | P90 TTFT (ms) | P95 TTFT (ms) | P99 TTFT (ms) | P90 TPOT (ms) | P95 TPOT (ms) | P99 TPOT (ms) |
|------------------|---------------|---------------|---------------|---------------|---------------|---------------|
| TBD              | TBD           | TBD           | TBD           | TBD           | TBD           | TBD           |

### 4.2 PD分离服务配置 (P2P NCCL)
| 请求速率 (req/s) | P90 TTFT (ms) | P95 TTFT (ms) | P99 TTFT (ms) | P90 TPOT (ms) | P95 TPOT (ms) | P99 TPOT (ms) |
|------------------|---------------|---------------|---------------|---------------|---------------|---------------|
| TBD              | TBD           | TBD           | TBD           | TBD           | TBD           | TBD           |

## 5. 结论

（由于目前还没有数据，无法得出结论。在实际测试完成后，这里将包含对结果的分析和结论。）
