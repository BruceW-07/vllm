# vLLM PD分离服务性能测试报告

## 1. 简介

本报告记录了使用 P2P NCCL 进行 KV 传输的 vLLM PD分离服务实现的性能测试。测试比较了PD合并服务配置与使用 P2P NCCL KV 传输的PD分离服务配置之间的性能。

## 1.1 性能指标说明

### 1.1.1 核心延迟指标
- **TTFT (Time To First Token)**: 从请求发送到接收到第一个token的时间，衡量服务响应速度
- **TPOT (Time Per Output Token)**: 每个输出token的平均生成时间，衡量解码阶段的效率
- **E2E Latency (End-to-End Latency)**: 从请求发送到完成整个响应的端到端延迟

### 1.1.2 延迟分解组件
**E2E延迟分解**采用简化的两组件模型：
- **TTFT (Time To First Token)**: 从请求发送到接收第一个token的完整时间
- **剩余Tokens解码时间**: 从第一个token到完成所有输出tokens的解码时间

**TTFT内部分解**（详细分析用）包含以下组件：
- **Prefill Queuing**: 请求在Prefill队列中的等待时间
- **Prefill Execution**: Prefill阶段的实际执行时间（处理输入tokens并生成KV cache）
- **KV Transfer**: KV cache从Prefill服务器传输到Decode服务器的时间（仅PD分离架构）
- **Decoding Queueing**: 请求在Decoding队列中的等待时间
- **First Token Decoding**: 生成第一个输出token的解码执行时间

### 1.1.3 百分位数指标
- **P90**: 90%的请求延迟低于此值
- **P95**: 95%的请求延迟低于此值
- **P99**: 99%的请求延迟低于此值

### 1.1.4 SLO达成率指标
SLO (Service Level Objective) 达成率衡量满足特定延迟目标的请求百分比：
- **TTFT SLO**: 满足TTFT延迟限制的请求百分比（默认限制：400ms）
- **TPOT SLO**: 满足TPOT延迟限制的请求百分比（默认限制：40ms）
- **Both SLO**: 同时满足TTFT和TPOT延迟限制的请求百分比

### 1.1.5 吞吐量指标
- **Request Rate**: 每秒处理的请求数量（req/s）
- **Per-GPU Rate**: 每个GPU的请求处理速率，用于不同GPU配置之间的公平比较

## 2. 实验设置

### 2.1 硬件配置
- GPU: 未指定

### 2.2 软件配置
- vLLM 版本: 未指定
- 模型: Llama-3.1-8B-Instruct

### 2.3 PD合并服务配置
- prefix-caching: disabled
- tensor-parallel-size: 1
- gpu-memory-utilization: 0.9
- max-model-len: 10000
- max-num-batched-tokens: 10000
- max-num-seqs: 256
- dtype: float16
- seed: 1024

### 2.4 PD分离服务配置 (P2P NCCL)

#### 2.4.1 架构配置
- Architecture: 1 Prefill + 1 Decode (1P1D)
- Prefill GPUs: 0
- Decode GPUs: 1
- Prefill ports: 20003
- Decode ports: 20005
- Proxy service discovery port: 30001
- Proxy app port: 10001

#### 2.4.2 Prefill服务器配置
- prefix-caching: disabled
- tensor-parallel-size: 1
- gpu-memory-utilization: 0.9
- max-model-len: 10000
- max-num-batched-tokens: 10000
- max-num-seqs: 256
- dtype: float16
- seed: 1024

#### 2.4.3 Decode服务器配置
- prefix-caching: disabled
- tensor-parallel-size: 1
- gpu-memory-utilization: 0.7
- max-model-len: 10000
- max-num-batched-tokens: 10000
- max-num-seqs: 256
- dtype: float16
- seed: 1024

#### 2.4.4 KV传输配置
- kv_connector: P2pNcclConnector
- kv_role: kv_producer/kv_consumer
- kv_buffer_size: 1e1 (producer), 8e9 (consumer)
- nccl_num_channels: 16

## 2.5 测试方法说明

### 2.5.1 性能测试配置
- **度量百分位数**: P90, P95, P99 用于评估延迟分布
- **随机种子**: 1024 确保测试结果的可重现性
- **请求数量**: 请求速率 × 300，确保充足的样本量进行统计分析

### 2.5.2 架构对比
**PD合并服务配置**:
- 单一服务器处理Prefill和Decode阶段
- GPU数量: 1个
- 请求速率范围: 0.5-5.5 req/s

**PD分离服务配置 (P2P NCCL)**:
- 分离架构: 1个Prefill GPU + 1个Decode GPU (1P1D)
- KV cache通过P2P NCCL在GPU间传输
- 请求速率范围: 1-11 req/s (总速率，对应0.5-5.5 req/s per GPU)

### 2.5.3 数据集说明
- **Random-512-64**: 固定输入长度512 tokens，输出长度64 tokens，用于基准测试
- **ShareGPT**: 真实对话数据集，输入输出长度变化较大，模拟实际使用场景
- **GSM8K**: 数学推理任务，固定输出长度256 tokens
- **Human Eval**: 代码生成任务，固定输出长度256 tokens

## 3. 性能结果

### 3.1 Random-512-64 数据集

#### 3.1.1 测试参数
- dataset-name: random
- random-input-len: 512 tokens
- random-output-len: 64 tokens
- ignore-eos: enabled
- metric-percentiles: 90, 95, 99
- seed: 1024
- Request rates:
  - PD-Combined serving: 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5
  - PD-disaggregation serving: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
- Number of prompts: request_rate * 300

#### 3.1.2 E2E延迟分解分析

##### 3.1.2.1 PD分离服务配置 (P2P NCCL)
![E2E延迟分解 - PD分离](plots/random-512-64/p2p_nccl_latency_breakdown.png)

*此图展示了E2E延迟的两个主要组件：TTFT和剩余tokens解码时间的相对占比。*

#### 3.1.3 TTFT 分解分析

##### 3.1.3.1 PD分离服务配置 (P2P NCCL)
![TTFT 分解 - PD分离](plots/random-512-64/p2p_nccl_ttft_breakdown.png)

#### 3.1.4 延迟比较 (每 GPU 的请求速率)

![延迟比较 - P90](plots/random-512-64/latency_rps_per_gpu_comparison_p90.png)
![延迟比较 - P95](plots/random-512-64/latency_rps_per_gpu_comparison_p95.png)
![延迟比较 - P99](plots/random-512-64/latency_rps_per_gpu_comparison_p99.png)

#### 3.1.5 SLO 达成率比较

![SLO 达成率 - TTFT 和 TPOT](plots/random-512-64/slo_attainment_rps_per_gpu_comparison.png)

### 3.2 ShareGPT 数据集

#### 3.2.1 测试参数
- dataset-name: sharegpt
- ignore-eos: enabled
- metric-percentiles: 90, 95, 99
- seed: 1024
- Request rates:
  - PD-Combined serving: 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5
  - PD-disaggregation serving: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
- Number of prompts: request_rate * 300

#### 3.2.2 延迟分解分析

##### 3.2.2.1 PD分离服务配置 (P2P NCCL)
![延迟分解 - PD分离](plots/sharegpt/p2p_nccl_latency_breakdown.png)

#### 3.2.3 TTFT 分解分析

##### 3.2.3.1 PD分离服务配置 (P2P NCCL)
![TTFT 分解 - PD分离](plots/sharegpt/p2p_nccl_ttft_breakdown.png)

#### 3.2.4 延迟比较 (每 GPU 的请求速率)

![延迟比较 - P90](plots/sharegpt/latency_rps_per_gpu_comparison_p90.png)
![延迟比较 - P95](plots/sharegpt/latency_rps_per_gpu_comparison_p95.png)
![延迟比较 - P99](plots/sharegpt/latency_rps_per_gpu_comparison_p99.png)

#### 3.2.5 SLO 达成率比较

![SLO 达成率 - TTFT 和 TPOT](plots/sharegpt/slo_attainment_rps_per_gpu_comparison.png)

### 3.3 GSM8K 数据集

#### 3.3.1 测试参数
- dataset-name: gsm8k
- ignore-eos: enabled
- metric-percentiles: 90, 95, 99
- output-len: 256
- seed: 1024
- Request rates:
  - PD-Combined serving: 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5
  - PD-disaggregation serving: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
- Number of prompts: request_rate * 300

#### 3.3.2 延迟分解分析

##### 3.3.2.1 PD分离服务配置 (P2P NCCL)
![延迟分解 - PD分离](plots/gsm8k/p2p_nccl_latency_breakdown.png)

#### 3.3.3 TTFT 分解分析

##### 3.3.3.1 PD分离服务配置 (P2P NCCL)
![TTFT 分解 - PD分离](plots/gsm8k/p2p_nccl_ttft_breakdown.png)

#### 3.3.4 延迟比较 (每 GPU 的请求速率)

![延迟比较 - P90](plots/gsm8k/latency_rps_per_gpu_comparison_p90.png)
![延迟比较 - P95](plots/gsm8k/latency_rps_per_gpu_comparison_p95.png)
![延迟比较 - P99](plots/gsm8k/latency_rps_per_gpu_comparison_p99.png)

#### 3.3.5 SLO 达成率比较

![SLO 达成率 - TTFT 和 TPOT](plots/gsm8k/slo_attainment_rps_per_gpu_comparison.png)

### 3.4 Human Eval 数据集

#### 3.4.1 测试参数
- dataset-name: human_eval
- ignore-eos: enabled
- metric-percentiles: 90, 95, 99
- output-len: 256
- seed: 1024
- Request rates:
  - PD-Combined serving: 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5
  - PD-disaggregation serving: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
- Number of prompts: request_rate * 300

#### 3.4.2 延迟分解分析

##### 3.4.2.1 PD分离服务配置 (P2P NCCL)
![延迟分解 - PD分离](plots/human_eval/p2p_nccl_latency_breakdown.png)

#### 3.4.3 TTFT 分解分析

##### 3.4.3.1 PD分离服务配置 (P2P NCCL)
![TTFT 分解 - PD分离](plots/human_eval/p2p_nccl_ttft_breakdown.png)

#### 3.4.4 延迟比较 (每 GPU 的请求速率)

![延迟比较 - P90](plots/human_eval/latency_rps_per_gpu_comparison_p90.png)
![延迟比较 - P95](plots/human_eval/latency_rps_per_gpu_comparison_p95.png)
![延迟比较 - P99](plots/human_eval/latency_rps_per_gpu_comparison_p99.png)

#### 3.4.5 SLO 达成率比较

![SLO 达成率 - TTFT 和 TPOT](plots/human_eval/slo_attainment_rps_per_gpu_comparison.png)
