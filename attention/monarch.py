import torch
from einops import rearrange


def _initial_right_query(q, key_rows, q_init):
    # q is [batch, query_outer, query_row, query_column, head, dim].
    # For default Wan 480p/81-frame generation: [B, 21, 30, 52, H, D].
    q_init = (q_init or "ith").lower()

    if q_init in {"identity", "ith", "i"}:
        if q.size(2) != key_rows:
            raise ValueError("q_init=ith needs query_row and key_row to have the same size.")
        return q

    if q_init in {"uniform", "mean", "avg", "average"}:
        return q.mean(dim=2, keepdim=True).expand(-1, -1, key_rows, -1, -1, -1)

    if q_init in {"first", "1st"}:
        return q[:, :, :1].expand(-1, -1, key_rows, -1, -1, -1)

    raise ValueError(f"unsupported q_init={q_init!r}; expected mean, 1st, or ith.")


def _to_monarch_blocks(x, f_tied, h_reduce, w_reduce, height, width):
    """[B, F*H*W, heads, dim] -> [B, outer, row, col, heads, dim]."""
    batch, seq_len, heads, dim = x.shape
    if height % h_reduce or width % w_reduce:
        raise ValueError("height/width must be divisible by h_reduce/w_reduce.")
    if seq_len % (height * width):
        raise ValueError("sequence length must be a multiple of height * width.")

    frames = seq_len // (height * width)
    if frames % f_tied:
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
    """[B, outer, row, col, heads, dim] -> [B, F*H*W, heads, dim]."""
    return rearrange(
        x,
        "b (outer h_group w_group) (frame row) col head dim -> "
        "b (outer frame h_group row w_group col) head dim",
        frame=f_tied,
        h_group=h_reduce,
        w_group=w_reduce,
    ).contiguous()


def _one_step_monarch_chunk(q, k, v, scale, q_init):
    # q:   [B, A, I, J, H, D] = [batch, query_outer, query_row, query_column, head, dim]
    # k/v: [B, F, K, L, H, D] = [batch, key_outer, key_row, key_column, head, dim]
    # Default 480p/81-frame Wan after Monarch rearrange:
    #   q:   [B, 21, 30, 52, H, D]
    #   k/v: [B, 21, 30, 52, H, D]
    #
    # For each fixed (A, F, J, K), this approximates the dense [I x L]
    # attention tile as left_factor[:, None] * right_factor[None, :].
    scale_sqrt = scale**0.5
    q_scaled = q * scale_sqrt
    k_scaled = k * scale_sqrt
    key_outer, key_rows = k.size(1), k.size(2)

    right_query = _initial_right_query(q_scaled, key_rows, q_init)
    right_logits = torch.einsum("bakjhd,bfklhd->bhafkjl", right_query, k_scaled)
    right_logits = right_logits.float()
    right_logits = right_logits - right_logits.amax(dim=-1, keepdim=True)
    right_factor = torch.softmax(right_logits, dim=-1).to(q.dtype)

    key_summary = torch.einsum("bhafkjl,bfklhd->bafjkhd", right_factor, k_scaled)
    right_log_z = torch.logsumexp(right_logits, dim=-1, keepdim=True)
    left_correction = (
        right_factor * (right_logits - right_log_z)
    ).sum(dim=-1, keepdim=True).transpose(-2, -3)

    left_logits = torch.einsum("bafjkhd,baijhd->bhafjki", key_summary, q_scaled)
    left_logits = left_logits - left_correction
    left_factor = rearrange(left_logits, "b h a f j k i -> b h a j i (f k)")
    left_factor = torch.softmax(left_factor.float(), dim=-1).to(q.dtype)
    left_factor = rearrange(
        left_factor,
        "b h a j i (f k) -> b h a f j k i",
        f=key_outer,
        k=key_rows,
    )

    values_by_tile = torch.einsum("bhafkjl,bfklhe->bafjkhe", right_factor, v)
    return torch.einsum("bhafjki,bafjkhe->baijhe", left_factor, values_by_tile)


def _one_step_monarch(q, k, v, scale, query_outer_chunk, q_init):
    # q/k/v are already Monarch blocks:
    #   q:   [B, query_outer, query_row, query_column, head, dim]
    #   k/v: [B, key_outer, key_row, key_column, head, dim]
    out = torch.empty(
        q.size(0),
        q.size(1),
        q.size(2),
        q.size(3),
        v.size(-2),
        v.size(-1),
        device=v.device,
        dtype=v.dtype,
    )

    for start in range(0, q.size(1), query_outer_chunk):
        end = min(start + query_outer_chunk, q.size(1))
        out[:, start:end] = _one_step_monarch_chunk(
            q[:, start:end],
            k,
            v,
            scale,
            q_init,
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
    q_init=None,
    query_outer_chunk=None,
):
    """One-step Monarch self-attention for Wan video tokens."""
    # Incoming q/k/v from Wan self-attention:
    #   q/k/v: [batch, sequence, head, dim]
    # Default 480p/81-frame Wan:
    #   sequence = frame * height * width = 21 * 30 * 52 = 32760.
    # After _to_monarch_blocks:
    #   q/k/v: [batch, 21, 30, 52, head, dim]
    if q.size(1) != k.size(1) or k.size(1) != v.size(1):
        raise ValueError("Monarch attention expects self-attention q/k/v lengths to match.")

    if sm_scale is None:
        sm_scale = q.size(-1) ** -0.5
    query_outer_chunk = max(1, int(query_outer_chunk or 1))

    q_blocks = _to_monarch_blocks(q, f_tied, h_reduce, w_reduce, h, w)
    k_blocks = _to_monarch_blocks(k, f_tied, h_reduce, w_reduce, h, w)
    v_blocks = _to_monarch_blocks(v, f_tied, h_reduce, w_reduce, h, w)

    out = _one_step_monarch(
        q_blocks,
        k_blocks,
        v_blocks,
        scale=sm_scale,
        query_outer_chunk=query_outer_chunk,
        q_init=q_init,
    )
    return _from_monarch_blocks(out, f_tied, h_reduce, w_reduce)


__all__ = ["monarch_attn"]
