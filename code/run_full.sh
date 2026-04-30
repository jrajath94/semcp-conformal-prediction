#!/bin/bash
# Resilient orchestration: skips already-completed steps, isolates GPU
# contexts between vLLM and HuggingFace stages.
set -uo pipefail
: "${HF_TOKEN:?HF_TOKEN env var must be set (export HF_TOKEN=hf_xxx before running)}"
export HF_TOKEN
export VLLM_USE_DEEP_GEMM=0
export TOKENIZERS_PARALLELISM=false

MODEL=Qwen/Qwen2.5-7B-Instruct
N_EXAMPLES=500
K_SAMPLES=10
MAX_NEW=48
DATA=/workspace/semcp/data
RESULTS=/workspace/semcp/results
LOGS=/workspace/semcp/logs

mkdir -p "$DATA" "$RESULTS" "$LOGS"
cd /workspace/semcp/code

run_dataset () {
  DS="$1"

  if [ -s "$DATA/${DS}_raw.json" ]; then
    echo "=== [$DS] skip generation (raw json present) ==="
  else
    echo "=== [$DS] generate sample pool ==="
    python3 -m experiments.generate_pool \
      --dataset "$DS" --n_examples "$N_EXAMPLES" --k_samples "$K_SAMPLES" \
      --max_new_tokens "$MAX_NEW" --model "$MODEL" \
      --output "$DATA/${DS}_raw.json" 2>&1 | tee "$LOGS/${DS}_gen.log"
  fi

  if [ -s "$DATA/${DS}_pool.json" ] && [ -s "$DATA/${DS}_pool.npz" ]; then
    echo "=== [$DS] skip build (pool present) ==="
  else
    echo "=== [$DS] build pools (NLI + embeddings) ==="
    # Run in a fresh process so vLLM's CUDA context is fully released.
    python3 -m experiments.build_pools \
      --input "$DATA/${DS}_raw.json" \
      --output_base "$DATA/${DS}_pool" 2>&1 | tee "$LOGS/${DS}_build.log"
  fi

  if [ -s "$RESULTS/$DS/summaries.json" ]; then
    echo "=== [$DS] skip methods (summaries present) ==="
  else
    echo "=== [$DS] run methods ==="
    python3 -m experiments.run_main \
      --pool_base "$DATA/${DS}_pool" \
      --output_dir "$RESULTS/$DS" \
      --alphas 0.10 --seeds 0 1 2 \
      --methods semcp conu safer lofreecp tecp 2>&1 | tee "$LOGS/${DS}_run.log"
  fi
}

run_dataset triviaqa
run_dataset squad

echo "=== ALL DONE ==="
ls -la "$RESULTS"/*/summaries.json
