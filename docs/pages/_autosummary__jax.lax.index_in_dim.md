- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.index_in_dim

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.index_in_dim.rst "Download source file")
-  .pdf

# jax.lax.index_in_dim

## Contents

- [`index_in_dim()`](#jax.lax.index_in_dim)

# jax.lax.index_in_dim[\#](#jax-lax-index-in-dim "Link to this heading")

jax.lax.index_in_dim(*operand*, *index*, *axis=0*, *keepdims=True*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/slicing.py#L1019-L1077)[\#](#jax.lax.index_in_dim "Link to this definition")  
Convenience wrapper around `lax.slice()` to perform int indexing.

This is effectively equivalent to `operand[...,`` ``index]` with the indexing applied on the specified axis.

Parameters:  
- **operand** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* *np.ndarray*) – an array to index.

- **index** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – integer index

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – the axis along which to apply the index (defaults to 0)

- **keepdims** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – boolean specifying whether the output array should preserve the rank of the input (default=True)

Returns:  
The subarray at the specified index.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Examples

Here is a one-dimensional example:

    >>> x = jnp.arange(4)
    >>> lax.index_in_dim(x, 2)
    Array([2], dtype=int32)

    >>> lax.index_in_dim(x, 2, keepdims=False)
    Array(2, dtype=int32)

Here are some two-dimensional examples:

    >>> x = jnp.arange(12).reshape(3, 4)
    >>> x
    Array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]], dtype=int32)

    >>> lax.index_in_dim(x, 1)
    Array([[4, 5, 6, 7]], dtype=int32)

    >>> lax.index_in_dim(x, 1, axis=1, keepdims=False)
    Array([1, 5, 9], dtype=int32)

See also

- [`jax.numpy.ndarray.at`](jax.numpy.ndarray.at.html#jax.numpy.ndarray.at "jax.numpy.ndarray.at")

- [`jax.lax.slice()`](jax.lax.slice.html#jax.lax.slice "jax.lax.slice")

- [`jax.lax.slice_in_dim()`](jax.lax.slice_in_dim.html#jax.lax.slice_in_dim "jax.lax.slice_in_dim")

- [`jax.lax.dynamic_index_in_dim()`](jax.lax.dynamic_index_in_dim.html#jax.lax.dynamic_index_in_dim "jax.lax.dynamic_index_in_dim")

[](jax.lax.imag.html "previous page")

previous

jax.lax.imag

[](jax.lax.index_take.html "next page")

next

jax.lax.index_take

Contents

- [`index_in_dim()`](#jax.lax.index_in_dim)

By The JAX authors

© Copyright 2024, The JAX Authors.\
