# -*- coding: utf-8 -*-
"""
PyTorch 与显卡的兼容性：避免「能 import CUDA、能分配显存，但无可用 kernel」
(例如 RTX 50 / Blackwell sm_120 与较旧的 cu124 预编译包)，从而在前端报
CUDA error: no kernel image is available for execution on the device.
"""
from __future__ import annotations

import os
import logging
from typing import Any

log = logging.getLogger(__name__)
_safety_applied: bool = False


def _blackwell_gpu_with_old_pytorch_wheels(torch) -> bool:
    """
    RTX 50 等 sm_12x 需要带 sm_120 的 PyTorch（多为 cu128+ 轮子）；旧 +cu124 等常有 no kernel image。
    在能分配显存但大量算子失败前，用版本串 + 算力做判定，比单次 conv 更稳。
    """
    try:
        if not torch.cuda.is_available():
            return False
        major, _minor = torch.cuda.get_device_capability(0)
    except Exception:  # noqa: BLE001
        return False
    if major < 12:
        return False
    ver = str(torch.__version__)
    if "cu128" in ver or "cu129" in ver or "cu130" in ver:
        return False
    if "+cu" in ver or "cuda" in ver.lower():
        # 有 CUDA 包但非 cu128 系，在 sm_12x 上高概率不兼容
        if any(x in ver for x in ("+cu124", "+cu126", "+cu121", "+cu118", "+cu120")):
            return True
    return False


def _strict_cuda_kernels_work(torch) -> bool:
    """在 GPU 上执行 matmul/conv/同步，不通过则整进程视为无 CUDA。"""
    try:
        t = torch.randn(4, 3, 64, 64, device="cuda")
        conv = torch.nn.Conv2d(3, 8, 3, padding=1).cuda()
        y = torch.nn.functional.relu(conv(t))
        _ = (y * 1.0).sum()
        torch.cuda.synchronize()
        return True
    except Exception as e:  # noqa: BLE001 - 需捕获 C++/CUDA 各类异常
        log.warning("严格 CUDA 自检未通过(将回退 CPU): %s", e)
        return False


def _patch_no_cuda(torch: Any) -> None:
    torch.cuda.is_available = lambda: False  # type: ignore[assignment, method-assign]
    if hasattr(torch, "set_default_device"):
        try:
            torch.set_default_device("cpu")
        except Exception:  # noqa: BLE001
            pass


def apply_torch_device_safety() -> None:
    """
    在导入业务模块、加载 YOLO/SAM 之前调用一次即可。
    - 环境变量 TORCH_FORCE_CPU=1：强制整进程不暴露 CUDA
    - 否则：若 PyTorch 与当前 GPU 不兼容(严格自检失败)，全进程不暴露 CUDA
    """
    global _safety_applied
    if _safety_applied:
        return
    _safety_applied = True

    force = os.environ.get("TORCH_FORCE_CPU", "").strip().lower() in ("1", "true", "yes", "on")
    if force:
        try:
            import torch

            _patch_no_cuda(torch)
            print("[torch_compat] TORCH_FORCE_CPU=1，使用 CPU 推理/训练")
        except ImportError:
            pass
        return

    try:
        import torch
    except ImportError:
        return

    if not torch.cuda.is_available():
        return

    if _blackwell_gpu_with_old_pytorch_wheels(torch):
        _patch_no_cuda(torch)
        print(
            "[torch_compat] 检测到 sm_12x(如 RTX50) GPU，但当前 PyTorch 为旧 CUDA 包(+cu124 等)，"
            "全进程已回退 CPU。要启用 GPU: pip 安装带 cu128 的 torch 见 scripts/install-pytorch-gpu-cu128.ps1 (Windows) 或"
            " pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 (Linux)。"
        )
        return

    try:
        mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if mem_gb < 2.0:
            print(f"[torch_compat] GPU 显存不足 ({mem_gb:.1f}GB < 2GB)，使用 CPU 模式")
            _patch_no_cuda(torch)
            return
    except Exception as e:  # noqa: BLE001
        log.debug("显存探针失败: %s", e)

    if not _strict_cuda_kernels_work(torch):
        _patch_no_cuda(torch)
        print(
            "[torch_compat] 当前 PyTorch 无法在本机 GPU 上执行算子(常见于 RTX50/Blackwell 而 torch 为旧 cu124 轮子)；"
            "全进程已回退到 CPU。升级至支持 sm_120 的 PyTorch(如 2.7+ cu128)后可恢复 GPU，"
            "或设 TORCH_FORCE_CPU=1 显式用 CPU。"
        )
        return


def model_device():
    """SAM/YOLO 等加载权重时使用。"""
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
