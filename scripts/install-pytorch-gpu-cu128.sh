#!/usr/bin/env bash
# 在 Linux venv 中安装 PyTorch cu128（RTX 50 / sm_12x 建议用，替代 +cu124）
# 用法:  PROJECT=/root/Vision_Platform bash scripts/install-pytorch-gpu-cu128.sh
set -euo pipefail
ROOT="${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
if [[ ! -d venv ]]; then
  echo "先创建 venv: python3 -m venv venv"
  exit 1
fi
./venv/bin/pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
./venv/bin/pip install -U pip
./venv/bin/pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/cu128"
./venv/bin/python -c "import torch; print('version', torch.__version__); print('cuda', torch.version.cuda if torch.cuda.is_available() else 'N/A', 'ok=', torch.cuda.is_available())"
