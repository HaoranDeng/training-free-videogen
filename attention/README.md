# Attention Notes

`core.py` is the dense baseline. It wraps PyTorch scaled dot-product attention
and keeps the same call shape as Wan:

```text
q, k, v: [batch, sequence, head, dim]
out:     [batch, sequence, head, dim]
```

`monarch.py` is the one-step MonarchAttention path. For default 480p/81-frame
Wan generation, the sequence is rearranged into:

```text
q: [batch, query_outer=21, query_row=30, query_column=52, head, dim]
k: [batch, key_outer=21, key_row=30, key_column=52, head, dim]
v: [batch, key_outer=21, key_row=30, key_column=52, head, dim]
```

Each fixed `(query_outer, key_outer, query_column, key_row)` tile is a
`30 queries x 52 keys` attention block. Monarch approximates that tile with
a rank-1 product:

```text
left_factor:  [30]
right_factor: [52]
tile ~= left_factor[:, None] * right_factor[None, :]
```

The only approximation step is:

```text
initialize right query -> right_factor -> left_factor -> value aggregation
```

The easiest experimental hook is `q_init` in `_initial_right_query`:

- `ith`: key row `i` initializes with query row `i`
- `mean`: every key row initializes with the mean query
- `1st`: every key row initializes with the first query row
