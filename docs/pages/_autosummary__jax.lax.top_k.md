- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.top_k

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.top_k.rst "Download source file")
-  .pdf

# jax.lax.top_k

## Contents

- [`top_k()`](#jax.lax.top_k)

# jax.lax.top_k[\#](#jax-lax-top-k "Link to this heading")

jax.lax.top_k(*operand*, *k*, *\**, *axis=-1*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L3547-L3587)[\#](#jax.lax.top_k "Link to this definition")  
Returns top `k` values and their indices along the specified axis of `operand`.

Parameters:  
- **operand** (*ArrayLike*) – N-dimensional array of non-complex type.

- **k** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – integer specifying the number of top entries.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – optional integer specifying the axis along which to compute the top `k` entries. Default is -1, indicating the last axis.

Returns:  
A tuple `(values,`` ``indices)` where

- `values` is an array containing the top k values along the last axis.

- `indices` is an array containing the indices corresponding to values.

Return type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\]

`values[...,`` ``i,`` ``...]` is the `i`-th largest entry in `operand` along the specified axis, and its index is `indices[...,`` ``i,`` ``...]`.

If two elements are equal, the lower-index element appears first.

See also

- [`jax.lax.approx_max_k()`](jax.lax.approx_max_k.html#jax.lax.approx_max_k "jax.lax.approx_max_k")

- [`jax.lax.approx_min_k()`](jax.lax.approx_min_k.html#jax.lax.approx_min_k "jax.lax.approx_min_k")

Examples

Find the largest three values, and their indices, within an array:

    >>> x = jnp.array([9., 3., 6., 4., 10.])
    >>> values, indices = jax.lax.top_k(x, 3)
    >>> values
    Array([10.,  9.,  6.], dtype=float32)
    >>> indices
    Array([4, 0, 2], dtype=int32)

[](jax.lax.tile.html "previous page")

previous

jax.lax.tile

[](jax.lax.transpose.html "next page")

next

jax.lax.transpose

Contents

- [`top_k()`](#jax.lax.top_k)

By The JAX authors

© Copyright 2024, The JAX Authors.\
