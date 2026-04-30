#!/bin/bash
# All-in-one bootstrap: install + run + leave artifacts on /workspace volume.
# Designed to survive SSH disconnects (uses nohup-friendly design).
set -uo pipefail
exec > >(tee -a /workspace/semcp/logs/bootstrap.log) 2>&1
echo "=== Bootstrap started at $(date) ==="

mkdir -p /workspace/semcp/{data,logs,results,code}
cd /workspace/semcp

echo "=== Step 1: install python deps ==="
if ! python3 -c "import vllm" 2>/dev/null; then
  pip install --no-cache-dir \
    vllm transformers sentence-transformers datasets accelerate \
    huggingface_hub scikit-learn matplotlib seaborn 2>&1 | tail -3
else
  echo "vllm already installed"
fi
python3 -c "import vllm, transformers, sentence_transformers, datasets; \
print(f'vllm={vllm.__version__}, transformers={transformers.__version__}')"

echo "=== Step 2: launch run_full.sh ==="
chmod +x /workspace/semcp/code/run_full.sh
bash /workspace/semcp/code/run_full.sh
echo "=== Bootstrap finished at $(date) ==="
