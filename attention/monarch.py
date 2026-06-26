import torch
from einops import rearrange


def _select_merge_queries(q, key_rows, q_init):
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


def _merge_key_column_tokens(merge_query, k, v):
    # For each fixed (query_outer, key_outer, query_column, key_row), merge
    # the key_column tokens into one merged key token and one merged value token.
    merge_logits = torch.einsum("bakjhd,bfklhd->bhafkjl", merge_query, k)
    merge_logits = merge_logits.float()
    merge_logits = merge_logits - merge_logits.amax(dim=-1, keepdim=True)
    merge_weights = torch.softmax(merge_logits, dim=-1).to(k.dtype)

    merged_keys = torch.einsum("bhafkjl,bfklhd->bafjkhd", merge_weights, k)
    merged_values = torch.einsum("bhafkjl,bfklhe->bafjkhe", merge_weights, v)

    # Correction term from the merge softmax normalizer. It keeps this rewrite
    # exactly equivalent to the previous one-step Monarch implementation.
    merge_log_z = torch.logsumexp(merge_logits, dim=-1, keepdim=True)
    routing_bias = (
        merge_weights * (merge_logits - merge_log_z)
    ).sum(dim=-1, keepdim=True).transpose(-2, -3)
    return merged_keys, merged_values, routing_bias


def _route_queries_to_merged_tokens(q, merged_keys, routing_bias, key_outer, key_rows):
    # Each query row attends to all merged tokens indexed by key_outer x key_row.
    routing_logits = torch.einsum("bafjkhd,baijhd->bhafjki", merged_keys, q)
    routing_logits = routing_logits - routing_bias
    routing_weights = rearrange(routing_logits, "b h a f j k i -> b h a j i (f k)")
    routing_weights = torch.softmax(routing_weights.float(), dim=-1).to(q.dtype)
    return rearrange(
        routing_weights,
        "b h a j i (f k) -> b h a f j k i",
        f=key_outer,
        k=key_rows,
    )


def _one_step_token_merging_chunk(q, k, v, scale, q_init):
    # q:   [B, A, I, J, H, D] = [batch, query_outer, query_row, query_column, head, dim]
    # k/v: [B, F, K, L, H, D] = [batch, key_outer, key_row, key_column, head, dim]
    # Default 480p/81-frame Wan after Monarch rearrange:
    #   q:   [B, 21, 30, 52, H, D]
    #   k/v: [B, 21, 30, 52, H, D]
    #
    # For each fixed (A, F, J, K), key-column tokens L are merged into one
    # token. Query rows I are then routed to those merged tokens.
    scale_sqrt = scale**0.5
    q_scaled = q * scale_sqrt
    k_scaled = k * scale_sqrt
    key_outer, key_rows = k.size(1), k.size(2)

    merge_query = _select_merge_queries(q_scaled, key_rows, q_init)
    merged_keys, merged_values, routing_bias = _merge_key_column_tokens(
        merge_query,
        k_scaled,
        v,
    )
    routing_weights = _route_queries_to_merged_tokens(
        q_scaled,
        merged_keys,
        routing_bias,
        key_outer,
        key_rows,
    )
    return torch.einsum("bhafjki,bafjkhe->baijhe", routing_weights, merged_values)


def _token_merging_attention(q, k, v, scale, query_outer_chunk, q_init):
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
        out[:, start:end] = _one_step_token_merging_chunk(
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
    """One-step Monarch self-attention written as token merging plus routing."""
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

    out = _token_merging_attention(
        q_blocks,
        k_blocks,
        v_blocks,
        scale=sm_scale,
        query_outer_chunk=query_outer_chunk,
        q_init=q_init,
    )
    return _from_monarch_blocks(out, f_tied, h_reduce, w_reduce)


__all__ = ["monarch_attn"]
