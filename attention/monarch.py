import os

import torch
from einops import rearrange


def _random_q_indices(num_choices, num_positions, device, random_seed):
    if random_seed is None:
        return torch.randint(num_choices, (num_positions,), device=device)

    try:
        gen = torch.Generator(device=device)
    except (TypeError, RuntimeError):
        gen = torch.Generator()
    gen.manual_seed(int(random_seed))
    try:
        return torch.randint(
            num_choices, (num_positions,), device=device, generator=gen
        )
    except RuntimeError:
        return torch.randint(num_choices, (num_positions,), generator=gen).to(device)


def _initial_r_query(q, k_size, q_init, random_seed):
    # q is [batch, query_outer, query_row, query_column, head, dim].
    #   e.g. [batch, 21, (30), 52, head, dim] for default 480p/81-frame inference.
    # k is [batch, key_outer, key_row, key_column, head, dim] in the caller.
    #   e.g. [batch, 21, 30, (52), head, dim] for default 480p/81-frame inference.
    # Parentheses mark default cross tile axes: query_row x key_column.
    # query_column=52 and key_row=30 index different rank-1 tiles.
    # q_init variants select along dim=2, the query_row/key_row axis.
    q_init = q_init.lower()
    if q_init in {"identity", "ith", "i"}:
        if q.size(2) != k_size:
            raise ValueError(
                "q_init=ith requires matched query/key block axes; use mean, "
                "1st, or random for row_row/col_col tile axes."
            )
        return q
    if q_init in {"uniform", "mean", "avg", "average"}:
        return q.mean(dim=2, keepdim=True).expand(-1, -1, k_size, -1, -1, -1)
    if q_init in {"first", "1st"}:
        return q[:, :, :1].expand(-1, -1, k_size, -1, -1, -1)
    if q_init == "random":
        idx = _random_q_indices(q.size(2), k_size, q.device, random_seed)
        return q.index_select(2, idx)
    raise ValueError(
        f"unsupported q_init={q_init!r}; expected mean, random, 1st, or ith."
    )


def _get_rearrange_fns(x, f_tied, h_reduce, w_reduce, h, w):
    b, _, nh, d = x.shape

    def rearrange_fn(x):
        x = x.view(
            b,
            -1,
            f_tied,
            h_reduce,
            h // h_reduce,
            w_reduce,
            w // w_reduce,
            nh,
            d,
        )
        return rearrange(x, "b a f c i e j h d -> b (a c e) (f i) j h d")

    def return_fn(x):
        return rearrange(
            x,
            "b (a c e) (f i) j h d -> b (a f c i e j) h d",
            c=h_reduce,
            e=w_reduce,
            f=f_tied,
        )

    return rearrange_fn, return_fn


def _normalize_tile_axes(tile_axes):
    tile_axes = tile_axes.lower().replace("-", "_")
    if tile_axes in {"cross", "row_col", "query_row_key_col"}:
        return "cross"
    if tile_axes in {"row_row", "rows"}:
        return "row_row"
    if tile_axes in {"col_col", "column_column", "columns"}:
        return "col_col"
    raise ValueError(
        f"unsupported tile_axes={tile_axes!r}; expected cross, row_row, or col_col."
    )


def _run_tiled_monarch_attention(
    q_blocks,
    k_blocks,
    v_blocks,
    scale,
    query_outer_chunk,
    q_init,
    random_seed,
    num_iters,
    tile_axes,
):
    if tile_axes == "cross":
        return _monarch_attention_blocks(
            q_blocks,
            k_blocks,
            v_blocks,
            scale,
            query_outer_chunk,
            q_init,
            random_seed,
            num_iters,
        )

    if tile_axes == "row_row":
        out = _monarch_attention_blocks(
            q_blocks.transpose(2, 3).contiguous(),
            k_blocks,
            v_blocks,
            scale,
            query_outer_chunk,
            q_init,
            random_seed,
            num_iters,
        )
        return out.transpose(2, 3).contiguous()

    if tile_axes == "col_col":
        return _monarch_attention_blocks(
            q_blocks,
            k_blocks.transpose(2, 3).contiguous(),
            v_blocks.transpose(2, 3).contiguous(),
            scale,
            query_outer_chunk,
            q_init,
            random_seed,
            num_iters,
        )

    raise AssertionError(f"unreachable tile_axes={tile_axes!r}")


def _monarch_attention_chunk(q, k, v, scale, q_init, random_seed, num_iters):
    if num_iters < 1:
        raise ValueError("num_iters must be >= 1.")

    sm_scale_sqrt = scale**0.5
    q_scaled = q * sm_scale_sqrt
    k_scaled = k * sm_scale_sqrt
    a_r = _initial_r_query(q_scaled, k_scaled.size(2), q_init, random_seed)
    c_r = torch.ones(
        (
            q.size(0),
            q.size(4),
            q.size(1),
            k.size(1),
            k.size(2),
            q.size(3),
            1,
        ),
        device=q.device,
        dtype=q.dtype,
    )

    for step in range(num_iters):
        if a_r.dim() == 6:
            r_logits = torch.einsum("bakjhd,bfklhd->bhafkjl", a_r, k_scaled)
        else:
            r_logits = torch.einsum("bafkjhd,bfklhd->bhafkjl", a_r, k_scaled)
        r_logits = r_logits.float() * (1.0 / (c_r + 1e-6)).clamp_max(1e4)
        r_logits = r_logits - r_logits.amax(dim=-1, keepdim=True)
        r = torch.softmax(r_logits, dim=-1).to(q.dtype)

        a_l = torch.einsum("bhafkjl,bfklhd->bafjkhd", r, k_scaled)
        logz = torch.logsumexp(r_logits, dim=-1, keepdim=True)
        c_l = (r * (r_logits - logz)).sum(dim=-1, keepdim=True).transpose(-2, -3)

        l_logits = torch.einsum("bafjkhd,baijhd->bhafjki", a_l, q_scaled) - c_l
        l = rearrange(l_logits, "b h a f j k i -> b h a j i (f k)")
        l = torch.softmax(l.float(), dim=-1).to(q.dtype)
        l = rearrange(
            l,
            "b h a j i (f k) -> b h a f j k i",
            f=k.size(1),
            k=k.size(2),
        )

        if step == num_iters - 1:
            y = torch.einsum("bhafkjl,bfklhe->bafjkhe", r, v)
            return torch.einsum("bhafjki,bafjkhe->baijhe", l, y)

        a_r = torch.einsum("bhafjki,baijhd->bafkjhd", l, q_scaled)
        c_r = l.sum(dim=-1, dtype=torch.float32).unsqueeze(-1).transpose(-2, -3)


def _monarch_attention_blocks(
    q, k, v, scale, query_outer_chunk, q_init, random_seed, num_iters
):
    b, outer_q, inner_i, inner_j, num_heads, value_dim = (
        q.size(0),
        q.size(1),
        q.size(2),
        q.size(3),
        v.size(-2),
        v.size(-1),
    )
    out = torch.empty(
        b,
        outer_q,
        inner_i,
        inner_j,
        num_heads,
        value_dim,
        device=v.device,
        dtype=v.dtype,
    )

    for start in range(0, outer_q, query_outer_chunk):
        end = min(start + query_outer_chunk, outer_q)
        out[:, start:end] = _monarch_attention_chunk(
            q[:, start:end], k, v, scale, q_init, random_seed, num_iters
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
    block_causal_size=None,
    num_iters=1,
    q_init=None,
    random_seed=None,
    query_outer_chunk=None,
    tile_axes=None,
):
    b, qs, nh, d = q.shape
    ks = k.shape[1]
    if sm_scale is None:
        sm_scale = d**-0.5

    q_init = os.environ.get("MONARCH_Q_INIT", q_init or "ith")
    tile_axes = _normalize_tile_axes(
        os.environ.get("MONARCH_TILE_AXES", tile_axes or "cross")
    )
    if "MONARCH_RANDOM_SEED" in os.environ:
        random_seed = int(os.environ["MONARCH_RANDOM_SEED"])
    if query_outer_chunk is None:
        query_outer_chunk = int(os.environ.get("MONARCH_QUERY_OUTER_CHUNK", "1"))
    query_outer_chunk = max(1, int(query_outer_chunk))

    if block_causal_size is not None:
        block_tokens = f_tied * h * w
        if qs != ks:
            raise ValueError("causal Monarch attention requires equal q/k lengths.")
        if qs % block_causal_size or block_causal_size % block_tokens:
            raise ValueError("block_causal_size must align with Monarch block sizes.")
        chunks = []
        for end in range(block_causal_size, qs + 1, block_causal_size):
            start = end - block_causal_size
            chunks.append(
                monarch_attn(
                    q[:, start:end],
                    k[:, :end],
                    v[:, :end],
                    f_tied,
                    h_reduce,
                    w_reduce,
                    h,
                    w,
                    sm_scale=sm_scale,
                    num_iters=num_iters,
                    q_init=q_init,
                    random_seed=random_seed,
                    query_outer_chunk=query_outer_chunk,
                    tile_axes=tile_axes,
                )
            )
        return torch.cat(chunks, dim=1)

    rearrange_fn, return_fn = _get_rearrange_fns(q, f_tied, h_reduce, w_reduce, h, w)
    q_blocks = rearrange_fn(q).contiguous()
    k_blocks = rearrange_fn(k).contiguous()
    v_blocks = rearrange_fn(v).contiguous()

    out = _run_tiled_monarch_attention(
        q_blocks,
        k_blocks,
        v_blocks,
        sm_scale,
        query_outer_chunk,
        q_init,
        random_seed,
        num_iters,
        tile_axes,
    )
    return return_fn(out)


__all__ = ["monarch_attn"]
