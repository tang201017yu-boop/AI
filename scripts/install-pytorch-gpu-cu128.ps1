#Requires -Version 5.1
# 在 Windows 上为当前 venv 安装带 CUDA 12.8 的 PyTorch（RTX 50 / sm_120 等需 cu128+）
# 用法: 在仓库根目录  .\scripts\install-pytorch-gpu-cu128.ps1
$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Pip = Join-Path $Root "venv\Scripts\pip.exe"
if (-not (Test-Path $Pip)) { Write-Error "未找到 venv，请先: python -m venv venv" }
Write-Host "==> 卸载旧 torch 三件套(若存在)..."
& $Pip uninstall -y torch torchvision torchaudio 2>$null
Write-Host "==> 从 PyTorch 官方源安装 cu128 轮子(体积大,需数分钟)..."
& $Pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/cu128"
Write-Host "==> 验证:"
& (Join-Path $Root "venv\Scripts\python.exe") -c "import torch; print('torch', torch.__version__); print('cuda.is_available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
