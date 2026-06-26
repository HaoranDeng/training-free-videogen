"""
RMSNorm used in Q/K projections: FlashInfer on NVIDIA CUDA, PyTorch fallback on ROCm.

FlashInfer's fused kernels expect ``device_type=cuda``; ROCm tensors are rejected at TVM dispatch.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from ._rocm_runtime import is_rocm

_USE_TORCH = os.environ.get("MONARCH_FORCE_TORCH_RMSNORM", "") == "1"


def _rmsnorm_torch(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
    *,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    x_f = input.float()
    inv_rms = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = (x_f * inv_rms).to(dtype=input.dtype) * weight
    if out is not None:
        out.copy_(y)
        return out
    return y


def rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    if _USE_TORCH or is_rocm():
        return _rmsnorm_torch(input, weight, eps, out, enable_pdl=enable_pdl)
    from flashinfer.norm import rmsnorm as _fi_rmsnorm

    return _fi_rmsnorm(input, weight, eps=eps, out=out, enable_pdl=enable_pdl)
