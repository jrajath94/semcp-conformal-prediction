#!/bin/bash
# Orchestrates the full SemCP experiment matrix on the H100 pod.
#   1. Generate K samples per question with Qwen2.5-7B for each dataset.
#   2. Build NLI partitions + sentence embeddings.
#   3. Run all 5 methods (semcp, conu, safer, lofreecp, tecp) at alpha=0.10
#      with 3 seeds, 50/50 cal/test split.
#
# Total expected wall clock on H100 80GB:
#   - Sampling: ~25 min/dataset (1000 q * 10 samples)
#   - Teacher-forced NLL: ~30 min/dataset
#   - NLI partition + embeddings: ~15 min/dataset
#   - Run main (5 methods x 3 seeds): ~5 min/dataset
#   Total: ~2.5 hours for 2 datasets.

set -euo pipefail
: "${HF_TOKEN:?HF_TOKEN env var must be set (export HF_TOKEN=hf_xxx before running)}"
export HF_TOKEN
export VLLM_USE_DEEP_GEMM=0
export TOKENIZERS_PARALLELISM=false

MODEL=Qwen/Qwen3-8B
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
  echo "=== [$DS] generate sample pool ==="
  python3 -m experiments.generate_pool \
    --dataset "$DS" --n_examples "$N_EXAMPLES" --k_samples "$K_SAMPLES" \
    --max_new_tokens "$MAX_NEW" --model "$MODEL" \
    --output "$DATA/${DS}_raw.json" 2>&1 | tee "$LOGS/${DS}_gen.log"

  echo "=== [$DS] build pools (NLI + embeddings) ==="
  python3 -m experiments.build_pools \
    --input "$DATA/${DS}_raw.json" \
    --output_base "$DATA/${DS}_pool" 2>&1 | tee "$LOGS/${DS}_build.log"

  echo "=== [$DS] run methods ==="
  python3 -m experiments.run_main \
    --pool_base "$DATA/${DS}_pool" \
    --output_dir "$RESULTS/$DS" \
    --alphas 0.10 --seeds 0 1 2 \
    --methods semcp conu safer lofreecp tecp 2>&1 | tee "$LOGS/${DS}_run.log"
}

run_dataset triviaqa
run_dataset squad

echo "=== ALL DONE ==="
ls -la "$RESULTS"/*/summaries.json
