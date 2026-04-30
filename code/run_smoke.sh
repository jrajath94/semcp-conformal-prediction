#!/bin/bash
set -u
: "${HF_TOKEN:?HF_TOKEN env var must be set}"
export HF_TOKEN
export VLLM_USE_DEEP_GEMM=0
export VLLM_DISABLE_COMPILE=1
export VLLM_USE_V1=1
cd /workspace/semcp/code
mkdir -p /workspace/semcp/data
pkill -f generate_pool 2>/dev/null || true
sleep 2
nohup python3 -m experiments.generate_pool \
  --dataset triviaqa \
  --n_examples 20 \
  --k_samples 5 \
  --max_new_tokens 32 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --output /workspace/semcp/data/smoke_triviaqa.json \
  > /workspace/semcp/data/smoke_log2.txt 2>&1 &
echo "started PID $!"
