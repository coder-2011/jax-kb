- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.dynamic_index_in_dim

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.dynamic_index_in_dim.rst "Download source file")
-  .pdf

# jax.lax.dynamic_index_in_dim

## Contents

- [`dynamic_index_in_dim()`](#jax.lax.dynamic_index_in_dim)

# jax.lax.dynamic_index_in_dim[\#](#jax-lax-dynamic-index-in-dim "Link to this heading")

jax.lax.dynamic_index_in_dim(*operand*, *index*, *axis=0*, *keepdims=True*, *\**, *allow_negative_indices=True*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/slicing.py#L1144-L1202)[\#](#jax.lax.dynamic_index_in_dim "Link to this definition")  
Convenience wrapper around dynamic_slice to perform int indexing.

This is roughly equivalent to the following Python indexing syntax applied along the specified axis: `operand[...,`` ``index]`.

Parameters:  
- **operand** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* *np.ndarray*) – an array to slice.

- **index** (*ArrayLike*) – the (possibly dynamic) start index

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – the axis along which to apply the slice (defaults to 0)

- **keepdims** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – boolean specifying whether the output should have the same rank as the input (default = True)

- **allow_negative_indices** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – boolean specifying whether negative indices are allowed. If true, negative indices are taken relative to the end of the array. If false, negative indices are out of bounds and the result is implementation defined.

Returns:  
An array containing the slice.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Examples

Here is a one-dimensional example:

    >>> x = jnp.arange(5)
    >>> dynamic_index_in_dim(x, 1)
    Array([1], dtype=int32)

    >>> dynamic_index_in_dim(x, 1, keepdims=False)
    Array(1, dtype=int32)

Here is a two-dimensional example:

    >>> x = jnp.arange(12).reshape(3, 4)
    >>> x
    Array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]], dtype=int32)

    >>> dynamic_index_in_dim(x, 1, axis=1, keepdims=False)
    Array([1, 5, 9], dtype=int32)

See also

- [`jax.numpy.ndarray.at`](jax.numpy.ndarray.at.html#jax.numpy.ndarray.at "jax.numpy.ndarray.at")

- [`jax.lax.index_in_dim()`](jax.lax.index_in_dim.html#jax.lax.index_in_dim "jax.lax.index_in_dim")

- [`jax.lax.dynamic_slice()`](jax.lax.dynamic_slice.html#jax.lax.dynamic_slice "jax.lax.dynamic_slice")

- [`jax.lax.dynamic_slice_in_dim()`](jax.lax.dynamic_slice_in_dim.html#jax.lax.dynamic_slice_in_dim "jax.lax.dynamic_slice_in_dim")

[](jax.lax.dot_general.html "previous page")

previous

jax.lax.dot_general

[](jax.lax.dynamic_slice.html "next page")

next

jax.lax.dynamic_slice

Contents

- [`dynamic_index_in_dim()`](#jax.lax.dynamic_index_in_dim)

By The JAX authors

© Copyright 2024, The JAX Authors.\
