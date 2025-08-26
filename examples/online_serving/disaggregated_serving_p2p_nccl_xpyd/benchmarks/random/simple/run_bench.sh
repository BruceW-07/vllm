#!/bin/bash

set -xe

REQUEST_RATE=(1 2 3 4 5 6 7 8 9 10 11)

mkdir -p results

for REQUEST_RATE in "${REQUEST_RATE[@]}"; do
    NUM_PROMPTS=$((REQUEST_RATE * 300))
    echo "Running benchmark with request rate: $REQUEST_RATE, total prompts: $NUM_PROMPTS"
    vllm bench serve \
        --backend vllm \
        --model "/workspace/models/Llama-3.1-8B-Instruct" \
        --endpoint /v1/completions \
        --dataset-name random  \
        --random-input-len 512 \
        --random-output-len 64 \
        --ignore-eos \
        --metric-percentiles "90,95,99" \
        --seed 1024 \
        --trust-remote-code \
        --request-rate $REQUEST_RATE \
        --num_prompt $NUM_PROMPTS \
        --save-result \
        --save-detailed \
        --result-dir ./results \
        --port 8027
        
    
    echo "Benchmark with request rate $REQUEST_RATE and $NUM_PROMPTS prompts completed."
    sleep 10
done