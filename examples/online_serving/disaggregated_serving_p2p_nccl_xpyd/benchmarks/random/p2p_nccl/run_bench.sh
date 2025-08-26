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
        --port 8028
        
    # Create logs directory if it doesn't exist
    mkdir -p logs
    
    # Save log lines containing "waiting for" string, creating separate log files for each REQUEST_RATE
    echo "Extracting 'waiting for' lines from logs for request rate $REQUEST_RATE..."
    if [ -f prefill1.log ]; then
        grep "waiting for" prefill1.log > logs/prefill1_queue_${REQUEST_RATE}.log
        echo "Saved logs/prefill1_queue_${REQUEST_RATE}.log"
        # Delete processed log files
        rm -f prefill1.log
    else
        echo "prefill1.log not found"
    fi
    
    if [ -f decode1.log ]; then
        grep "waiting for" decode1.log > logs/decode1_queue_${REQUEST_RATE}.log
        echo "Saved logs/decode1_queue_${REQUEST_RATE}.log"
        # Delete processed log files
        rm -f decode1.log
    else
        echo "decode1.log not found"
    fi
    
    echo "Benchmark with request rate $REQUEST_RATE and $NUM_PROMPTS prompts completed."
    sleep 10
done