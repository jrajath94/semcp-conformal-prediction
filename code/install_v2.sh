#!/bin/bash
# Cleaner v2 install: uninstall everything first, then install single set.
set -uo pipefail
exec > >(tee -a /workspace/semcp/logs/install.log) 2>&1
echo "=== Install v2 started at $(date) ==="

pip uninstall -y torch torchvision torchaudio vllm transformers tokenizers \
  sentence-transformers datasets accelerate huggingface_hub 2>&1 | tail -3 || true

# Pin known-good combo built against CUDA 12.4
pip install --no-cache-dir \
  "torch==2.4.0" "torchvision==0.19.0" \
  --index-url https://download.pytorch.org/whl/cu124

# vllm 0.6.3.post1 brings in compatible transformers (4.45.x) and tokenizers (0.20.x)
pip install --no-cache-dir "vllm==0.6.3.post1"

# Sentence-transformers 3.x is happy with transformers 4.45.x
pip install --no-cache-dir \
  "sentence-transformers==3.1.1" \
  "datasets==3.0.1" \
  "accelerate==0.34.2" \
  scikit-learn matplotlib seaborn pyarrow

python3 -c "
import torch, vllm, transformers, sentence_transformers, datasets
print(f'torch={torch.__version__} cuda={torch.version.cuda}')
print(f'vllm={vllm.__version__} transformers={transformers.__version__}')
print(f'gpu_available={torch.cuda.is_available()}, gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')
print('IMPORT_OK')
"
echo "=== Install v2 done at $(date) ==="
