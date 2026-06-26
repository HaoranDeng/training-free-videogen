# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import torch.nn.functional as F

__all__ = [
    'flash_attention',
    'attention',
]


def _build_sdpa_mask(
    b,
    lq,
    lk,
    device,
    q_lens=None,
    k_lens=None,
    causal=False,
    window_size=(-1, -1),
):
    allow = torch.ones(b, lq, lk, dtype=torch.bool, device=device)

    if k_lens is not None:
        k_valid = torch.arange(lk, device=device).unsqueeze(0) < k_lens.unsqueeze(1)
        allow = allow & k_valid.unsqueeze(1)

    if q_lens is not None:
        q_valid = torch.arange(lq, device=device).unsqueeze(0) < q_lens.unsqueeze(1)
        allow = allow & q_valid.unsqueeze(2)

    if causal:
        q_pos = torch.arange(lq, device=device).unsqueeze(1)
        k_pos = torch.arange(lk, device=device).unsqueeze(0)
        allow = allow & (k_pos <= q_pos)

    if window_size != (-1, -1):
        left, right = window_size
        q_pos = torch.arange(lq, device=device).unsqueeze(1)
        k_pos = torch.arange(lk, device=device).unsqueeze(0)
        allow = allow & (k_pos >= q_pos - left) & (k_pos <= q_pos + right)

    return allow.unsqueeze(1)


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, use the math SDPA backend for reproducibility.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    version:        Ignored (kept for API compatibility).
    """
    del version

    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes

    out_dtype = q.dtype
    b, lq, lk = q.size(0), q.size(1), k.size(1)

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    q = half(q)
    k = half(k)
    v = half(v)

    if q_scale is not None:
        q = q * q_scale

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    use_custom_mask = (
        q_lens is not None or k_lens is not None or window_size != (-1, -1)
        or (causal and lq != lk))
    if causal and not use_custom_mask:
        is_causal = True
        attn_mask = None
    else:
        is_causal = False
        if causal or q_lens is not None or k_lens is not None or window_size != (-1, -1):
            allow = _build_sdpa_mask(
                b,
                lq,
                lk,
                q.device,
                q_lens=q_lens,
                k_lens=k_lens,
                causal=causal,
                window_size=window_size,
            )
            attn_mask = torch.zeros(
                b, 1, lq, lk, device=q.device, dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(~allow, float('-inf'))
        else:
            attn_mask = None

    sdpa_kwargs = dict(
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=softmax_scale,
    )

    if deterministic and q.is_cuda:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            x = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, **sdpa_kwargs)
    else:
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, **sdpa_kwargs)

    x = x.transpose(1, 2).contiguous()
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    del fa_version
    return flash_attention(
        q=q,
        k=k,
        v=v,
        q_lens=q_lens,
        k_lens=k_lens,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        q_scale=q_scale,
        causal=causal,
        window_size=window_size,
        deterministic=deterministic,
        dtype=dtype,
    )
