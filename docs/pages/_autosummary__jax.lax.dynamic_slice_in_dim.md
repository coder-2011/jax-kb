- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.dynamic_slice_in_dim

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.dynamic_slice_in_dim.rst "Download source file")
-  .pdf

# jax.lax.dynamic_slice_in_dim

## Contents

- [`dynamic_slice_in_dim()`](#jax.lax.dynamic_slice_in_dim)

# jax.lax.dynamic_slice_in_dim[\#](#jax-lax-dynamic-slice-in-dim "Link to this heading")

jax.lax.dynamic_slice_in_dim(*operand*, *start_index*, *slice_size*, *axis=0*, *\**, *allow_negative_indices=True*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/slicing.py#L1079-L1142)[\#](#jax.lax.dynamic_slice_in_dim "Link to this definition")  
Convenience wrapper around `lax.dynamic_slice()` applied to one dimension.

This is roughly equivalent to the following Python indexing syntax applied along the specified axis: `operand[...,`` ``start_index:start_index`` ``+`` ``slice_size]`.

Parameters:  
- **operand** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* *np.ndarray*) – an array to slice.

- **start_index** (*ArrayLike*) – the (possibly dynamic) start index

- **slice_size** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – the static slice size

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – the axis along which to apply the slice (defaults to 0)

- **allow_negative_indices** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – boolean specifying whether negative indices are allowed. If true, negative indices are taken relative to the end of the array. If false, negative indices are out of bounds and the result is implementation defined.

Returns:  
An array containing the slice.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Examples

Here is a one-dimensional example:

    >>> x = jnp.arange(5)
    >>> dynamic_slice_in_dim(x, 1, 3)
    Array([1, 2, 3], dtype=int32)

Like jax.lax.dynamic_slice, out-of-bound slices will be clipped to the valid range:

    >>> dynamic_slice_in_dim(x, 4, 3)
    Array([2, 3, 4], dtype=int32)

Here is a two-dimensional example:

    >>> x = jnp.arange(12).reshape(3, 4)
    >>> x
    Array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]], dtype=int32)

    >>> dynamic_slice_in_dim(x, 1, 2, axis=1)
    Array([[ 1,  2],
           [ 5,  6],
           [ 9, 10]], dtype=int32)

See also

- [`jax.numpy.ndarray.at`](jax.numpy.ndarray.at.html#jax.numpy.ndarray.at "jax.numpy.ndarray.at")

- [`jax.lax.slice_in_dim()`](jax.lax.slice_in_dim.html#jax.lax.slice_in_dim "jax.lax.slice_in_dim")

- [`jax.lax.dynamic_slice()`](jax.lax.dynamic_slice.html#jax.lax.dynamic_slice "jax.lax.dynamic_slice")

- [`jax.lax.dynamic_index_in_dim()`](jax.lax.dynamic_index_in_dim.html#jax.lax.dynamic_index_in_dim "jax.lax.dynamic_index_in_dim")

[](jax.lax.dynamic_slice.html "previous page")

previous

jax.lax.dynamic_slice

[](jax.lax.dynamic_update_index_in_dim.html "next page")

next

jax.lax.dynamic_update_index_in_dim

Contents

- [`dynamic_slice_in_dim()`](#jax.lax.dynamic_slice_in_dim)

By The JAX authors

© Copyright 2024, The JAX Authors.\
