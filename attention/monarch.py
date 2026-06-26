import torch
from einops import rearrange


def _rand_indices(num_choices, num_indices, device, seed):
    if seed is None:
        return torch.randint(num_choices, (num_indices,), device=device)

    try:
        generator = torch.Generator(device=device)
    except (TypeError, RuntimeError):
        generator = torch.Generator()
    generator.manual_seed(int(seed))

    try:
        return torch.randint(
            num_choices,
            (num_indices,),
            device=device,
            generator=generator,
        )
    except RuntimeError:
        return torch.randint(num_choices, (num_indices,), generator=generator).to(device)


def _initial_right_query(q, key_rows, q_init, random_seed):
    # q is [batch, query_outer, query_row, query_column, head, dim].
    # For 480p/81-frame Wan inference this is [B, 21, 30, 52, H, D].
    #
    # A Monarch rank-1 tile fixes (query_outer, key_outer, query_column, key_row)
    # and approximates the [query_row x key_column] attention submatrix.
    # Therefore this initializer selects one query vector for each key_row.
    q_init = (q_init or "ith").lower()

    if q_init in {"identity", "ith", "i"}:
        if q.size(2) != key_rows:
            raise ValueError("q_init=ith needs query_row and key_row to have the same size.")
        return q

    if q_init in {"uniform", "mean", "avg", "average"}:
        return q.mean(dim=2, keepdim=True).expand(-1, -1, key_rows, -1, -1, -1)

    if q_init in {"first", "1st"}:
        return q[:, :, :1].expand(-1, -1, key_rows, -1, -1, -1)

    if q_init == "random":
        indices = _rand_indices(q.size(2), key_rows, q.device, random_seed)
        return q.index_select(2, indices)

    raise ValueError(f"unsupported q_init={q_init!r}; expected mean, random, 1st, or ith.")


def _to_monarch_blocks(x, f_tied, h_reduce, w_reduce, height, width):
    """Convert [B, F*H*W, heads, dim] to [B, outer, row, col, heads, dim]."""
    batch, seq_len, heads, dim = x.shape
    if height % h_reduce != 0 or width % w_reduce != 0:
        raise ValueError("height/width must be divisible by h_reduce/w_reduce.")
    if seq_len % (height * width) != 0:
        raise ValueError("sequence length must be a multiple of height * width.")

    frames = seq_len // (height * width)
    if frames % f_tied != 0:
        raise ValueError("number of frames must be divisible by f_tied.")

    x = x.view(
        batch,
        frames // f_tied,
        f_tied,
        h_reduce,
        height // h_reduce,
        w_reduce,
        width // w_reduce,
        heads,
        dim,
    )
    return rearrange(
        x,
        "b outer frame h_group row w_group col head dim -> "
        "b (outer h_group w_group) (frame row) col head dim",
    ).contiguous()


def _from_monarch_blocks(x, f_tied, h_reduce, w_reduce):
    """Convert [B, outer, row, col, heads, dim] back to [B, F*H*W, heads, dim]."""
    return rearrange(
        x,
        "b (outer h_group w_group) (frame row) col head dim -> "
        "b (outer frame h_group row w_group col) head dim",
        frame=f_tied,
        h_group=h_reduce,
        w_group=w_reduce,
    ).contiguous()


def _softmax_right(right_query, key, normalizer):
    # right_query is initially [B, A, K, J, H, D].
    # After the first iteration it becomes [B, A, F, K, J, H, D].
    if right_query.dim() == 6:
        logits = torch.einsum("bakjhd,bfklhd->bhafkjl", right_query, key)
    else:
        logits = torch.einsum("bafkjhd,bfklhd->bhafkjl", right_query, key)

    logits = logits.float() * (1.0 / (normalizer + 1e-6)).clamp_max(1e4)
    logits = logits - logits.amax(dim=-1, keepdim=True)
    return torch.softmax(logits, dim=-1).to(key.dtype), logits


def _softmax_left(logits, key_outer, key_rows, out_dtype):
    # Softmax over all key_outer x key_row choices for each query row.
    probs = rearrange(logits, "b h a f j k i -> b h a j i (f k)")
    probs = torch.softmax(probs.float(), dim=-1).to(out_dtype)
    return rearrange(
        probs,
        "b h a j i (f k) -> b h a f j k i",
        f=key_outer,
        k=key_rows,
    )


def _monarch_attention_chunk(q, k, v, scale, q_init, random_seed, num_iters):
    # q: [B, A, I, J, H, D]
    # k/v: [B, F, K, L, H, D]
    #
    # For each fixed (A, F, J, K), Monarch approximates the dense [I x L]
    # attention tile by a rank-1 product L_factor[:, None] * R_factor[None, :].
    if num_iters < 1:
        raise ValueError("num_iters must be >= 1.")

    scale_sqrt = scale**0.5
    q_scaled = q * scale_sqrt
    k_scaled = k * scale_sqrt

    batch, query_outer, query_rows, query_cols, heads, _ = q.shape
    key_outer, key_rows = k.size(1), k.size(2)

    right_query = _initial_right_query(q_scaled, key_rows, q_init, random_seed)
    right_normalizer = torch.ones(
        (batch, heads, query_outer, key_outer, key_rows, query_cols, 1),
        device=q.device,
        dtype=q.dtype,
    )

    for step in range(num_iters):
        right_factor, right_logits = _softmax_right(right_query, k_scaled, right_normalizer)

        key_summary = torch.einsum("bhafkjl,bfklhd->bafjkhd", right_factor, k_scaled)
        right_log_z = torch.logsumexp(right_logits, dim=-1, keepdim=True)
        left_correction = (
            right_factor * (right_logits - right_log_z)
        ).sum(dim=-1, keepdim=True).transpose(-2, -3)

        left_logits = torch.einsum("bafjkhd,baijhd->bhafjki", key_summary, q_scaled)
        left_factor = _softmax_left(
            left_logits - left_correction,
            key_outer=key_outer,
            key_rows=key_rows,
            out_dtype=q.dtype,
        )

        if step == num_iters - 1:
            values_by_tile = torch.einsum("bhafkjl,bfklhe->bafjkhe", right_factor, v)
            return torch.einsum("bhafjki,bafjkhe->baijhe", left_factor, values_by_tile)

        right_query = torch.einsum("bhafjki,baijhd->bafkjhd", left_factor, q_scaled)
        right_normalizer = (
            left_factor.sum(dim=-1, dtype=torch.float32)
            .unsqueeze(-1)
            .transpose(-2, -3)
        )


def _monarch_attention(q, k, v, scale, query_outer_chunk, q_init, random_seed, num_iters):
    batch, query_outer, query_rows, query_cols = q.shape[:4]
    heads, value_dim = v.size(-2), v.size(-1)
    out = torch.empty(
        batch,
        query_outer,
        query_rows,
        query_cols,
        heads,
        value_dim,
        device=v.device,
        dtype=v.dtype,
    )

    for start in range(0, query_outer, query_outer_chunk):
        end = min(start + query_outer_chunk, query_outer)
        out[:, start:end] = _monarch_attention_chunk(
            q[:, start:end],
            k,
            v,
            scale,
            q_init,
            random_seed,
            num_iters,
        )
    return out


def monarch_attn(
    q,
    k,
    v,
    f_tied,
    h_reduce,
    w_reduce,
    h,
    w,
    sm_scale=None,
    num_iters=1,
    q_init=None,
    random_seed=None,
    query_outer_chunk=None,
):
    """Monarch self-attention for Wan video tokens.

    This minimal repository supports the default non-causal Monarch tiling:
    [query_row x key_column] rank-1 tiles. The older experiment-only row_row,
    col_col, and causal-cache paths are intentionally not implemented here.
    """
    if q.size(1) != k.size(1) or k.size(1) != v.size(1):
        raise ValueError("minimal Monarch attention expects self-attention q/k/v lengths to match.")

    head_dim = q.size(-1)
    if sm_scale is None:
        sm_scale = head_dim**-0.5
    query_outer_chunk = max(1, int(query_outer_chunk or 1))

    q_blocks = _to_monarch_blocks(q, f_tied, h_reduce, w_reduce, h, w)
    k_blocks = _to_monarch_blocks(k, f_tied, h_reduce, w_reduce, h, w)
    v_blocks = _to_monarch_blocks(v, f_tied, h_reduce, w_reduce, h, w)

    out = _monarch_attention(
        q_blocks,
        k_blocks,
        v_blocks,
        scale=sm_scale,
        query_outer_chunk=query_outer_chunk,
        q_init=q_init,
        random_seed=random_seed,
        num_iters=num_iters,
    )
    return _from_monarch_blocks(out, f_tied, h_reduce, w_reduce)


__all__ = ["monarch_attn"]
