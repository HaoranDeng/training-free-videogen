# Attention Notes

`core.py` is the dense baseline. It wraps PyTorch scaled dot-product attention
and keeps the same call shape as Wan:

```text
q, k, v: [batch, sequence, head, dim]
out:     [batch, sequence, head, dim]
```

`monarch.py` is the one-step MonarchAttention path, written in a token-merging
style. For default 480p/81-frame Wan generation, the sequence is rearranged into:

```text
q: [batch, query_outer=21, query_row=30, query_column=52, head, dim]
k: [batch, key_outer=21, key_row=30, key_column=52, head, dim]
v: [batch, key_outer=21, key_row=30, key_column=52, head, dim]
```

Each fixed `(query_outer, key_outer, query_column, key_row)` tile is a
`30 queries x 52 keys` attention block. The code first merges the `52`
key-column tokens into one merged key/value token, then routes the `30`
query rows to those merged tokens.

```text
merge_weights:   [52]
routing_weights: [30]
tile ~= routing_weights[:, None] * merge_weights[None, :]
```

The only approximation step is:

```text
select merge query -> merge key/value tokens -> route queries -> aggregate values
```

The easiest experimental hook is `q_init` in `_select_merge_queries`:

- `ith`: key row `i` initializes with query row `i`
- `mean`: every key row initializes with the mean query
- `1st`: every key row initializes with the first query row
