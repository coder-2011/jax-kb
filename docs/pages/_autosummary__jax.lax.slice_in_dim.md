- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.slice_in_dim

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.slice_in_dim.rst "Download source file")
-  .pdf

# jax.lax.slice_in_dim

## Contents

- [`slice_in_dim()`](#jax.lax.slice_in_dim)

# jax.lax.slice_in_dim[\#](#jax-lax-slice-in-dim "Link to this heading")

jax.lax.slice_in_dim(*operand*, *start_index*, *limit_index*, *stride=1*, *axis=0*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/slicing.py#L944-L1017)[\#](#jax.lax.slice_in_dim "Link to this definition")  
Convenience wrapper around `lax.slice()` applying to only one dimension.

This is effectively equivalent to `operand[...,`` ``start_index:limit_index:stride]` with the indexing applied on the specified axis.

Parameters:  
- **operand** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* *np.ndarray*) – an array to slice.

- **start_index** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – an optional start index (defaults to zero)

- **limit_index** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – an optional end index (defaults to operand.shape\[axis\])

- **stride** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – an optional stride (defaults to 1)

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – the axis along which to apply the slice (defaults to 0)

Returns:  
An array containing the slice.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Examples

Here is a one-dimensional example:

    >>> x = jnp.arange(4)
    >>> lax.slice_in_dim(x, 1, 3)
    Array([1, 2], dtype=int32)

Here are some two-dimensional examples:

    >>> x = jnp.arange(12).reshape(4, 3)
    >>> x
    Array([[ 0,  1,  2],
           [ 3,  4,  5],
           [ 6,  7,  8],
           [ 9, 10, 11]], dtype=int32)

    >>> lax.slice_in_dim(x, 1, 3)
    Array([[3, 4, 5],
           [6, 7, 8]], dtype=int32)

    >>> lax.slice_in_dim(x, 1, 3, axis=1)
    Array([[ 1,  2],
           [ 4,  5],
           [ 7,  8],
           [10, 11]], dtype=int32)

See also

- [`jax.numpy.ndarray.at`](jax.numpy.ndarray.at.html#jax.numpy.ndarray.at "jax.numpy.ndarray.at")

- [`jax.lax.slice()`](jax.lax.slice.html#jax.lax.slice "jax.lax.slice")

- [`jax.lax.index_in_dim()`](jax.lax.index_in_dim.html#jax.lax.index_in_dim "jax.lax.index_in_dim")

- [`jax.lax.dynamic_slice_in_dim()`](jax.lax.dynamic_slice_in_dim.html#jax.lax.dynamic_slice_in_dim "jax.lax.dynamic_slice_in_dim")

[](jax.lax.slice.html "previous page")

previous

jax.lax.slice

[](jax.lax.sort.html "next page")

next

jax.lax.sort

Contents

- [`slice_in_dim()`](#jax.lax.slice_in_dim)

By The JAX authors

© Copyright 2024, The JAX Authors.\
