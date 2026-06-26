"""FlashInfer RMSNorm kernels read ``shared_memory_per_block_optin`` from
``torch.cuda.get_device_properties``. ROCm HIP builds omit this field; Fall back to
``shared_memory_per_block`` so FlashInfer can initialize."""

from __future__ import annotations

import torch

_PATCH_APPLIED = False


class _CudaDevicePropsWrapper:
    __slots__ = ("_props",)

    def __init__(self, props: object):
        object.__setattr__(self, "_props", props)

    def __getattr__(self, name: str):
        props = object.__getattribute__(self, "_props")
        if name == "shared_memory_per_block_optin":
            return props.shared_memory_per_block
        return getattr(props, name)

    def __repr__(self) -> str:
        return repr(object.__getattribute__(self, "_props"))


def ensure_patched_torch_cuda_device_properties() -> None:
    """Idempotent monkey-patch for ROCm / older PyTorch device property stubs."""
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return

    inner = torch.cuda.get_device_properties

    def wrapped(*args, **kwargs):
        props = inner(*args, **kwargs)
        if hasattr(props, "shared_memory_per_block_optin"):
            return props
        return _CudaDevicePropsWrapper(props)

    torch.cuda.get_device_properties = wrapped  # type: ignore[method-assign]
    _PATCH_APPLIED = True
