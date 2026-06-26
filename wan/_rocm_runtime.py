"""Detect ROCm / HIP PyTorch builds (FlashInfer CUDA kernels are not usable)."""

from __future__ import annotations

import torch


def is_rocm() -> bool:
    return bool(getattr(torch.version, "hip", None))
