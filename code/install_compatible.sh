#!/bin/bash
# Install vLLM stack compatible with CUDA 12.8 driver.
set -uo pipefail
exec > >(tee -a /workspace/semcp/logs/install.log) 2>&1
echo "=== Install started at $(date) ==="

# Uninstall any wrong-CUDA torch first
pip uninstall -y torch torchvision torchaudio vllm 2>&1 | tail -3 || true

# Install torch built against CUDA 12.4 (works with driver 12.8)
pip install --no-cache-dir torch==2.4.0 torchvision==0.19.0 \
  --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -3

# Install vllm version compatible with torch 2.4 (released Sep 2024)
pip install --no-cache-dir vllm==0.6.3.post1 2>&1 | tail -3

# Other deps (avoid auto-upgrading torch)
pip install --no-cache-dir --no-deps \
  transformers==4.45.2 sentence-transformers==3.1.1 datasets==3.0.1 \
  accelerate==0.34.2 huggingface_hub==0.25.2 2>&1 | tail -3
pip install --no-cache-dir tokenizers safetensors regex tqdm filelock packaging \
  pyyaml requests numpy scipy scikit-learn matplotlib seaborn pyarrow \
  fsspec aiohttp pandas xxhash multiprocess dill protobuf sentencepiece \
  einops 2>&1 | tail -3

python3 -c "
import torch, vllm, transformers, sentence_transformers, datasets
print(f'torch={torch.__version__}, cuda={torch.version.cuda}')
print(f'vllm={vllm.__version__}, transformers={transformers.__version__}')
print(f'gpu_available={torch.cuda.is_available()}')
print(f'gpu_name={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')
"
echo "=== Install done at $(date) ==="
