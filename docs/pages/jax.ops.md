- [](index.html)
- [API Reference](jax.html)
- `jax.ops` module

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](_sources/jax.ops.rst "Download source file")
-  .pdf

# jax.ops module

## Contents

- [Segment reduction operators](#segment-reduction-operators)

# `jax.ops` module[\#](#module-jax.ops "Link to this heading")

The functions `jax.ops.index_update`, `jax.ops.index_add`, etc., which were deprecated in JAX 0.2.22, have been removed. Please use the [`jax.numpy.ndarray.at`](_autosummary/jax.numpy.ndarray.at.html#jax.numpy.ndarray.at "jax.numpy.ndarray.at") property on JAX arrays instead.

## Segment reduction operators[\#](#segment-reduction-operators "Link to this heading")

|  |  |
|----|----|
| [`segment_max`](_autosummary/jax.ops.segment_max.html#jax.ops.segment_max "jax.ops.segment_max")(data, segment_ids\[, ...\]) | Computes the maximum within segments of an array. |
| [`segment_min`](_autosummary/jax.ops.segment_min.html#jax.ops.segment_min "jax.ops.segment_min")(data, segment_ids\[, ...\]) | Computes the minimum within segments of an array. |
| [`segment_prod`](_autosummary/jax.ops.segment_prod.html#jax.ops.segment_prod "jax.ops.segment_prod")(data, segment_ids\[, ...\]) | Computes the product within segments of an array. |
| [`segment_sum`](_autosummary/jax.ops.segment_sum.html#jax.ops.segment_sum "jax.ops.segment_sum")(data, segment_ids\[, ...\]) | Computes the sum within segments of an array. |

[](_autosummary/jax.nn.log1mexp.html "previous page")

previous

jax.nn.log1mexp

[](_autosummary/jax.ops.segment_max.html "next page")

next

jax.ops.segment_max

Contents

- [Segment reduction operators](#segment-reduction-operators)

By The JAX authors

© Copyright 2024, The JAX Authors.\
