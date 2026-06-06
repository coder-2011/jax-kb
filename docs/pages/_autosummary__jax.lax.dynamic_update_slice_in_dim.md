- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.dynamic_update_slice_in_dim

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.dynamic_update_slice_in_dim.rst "Download source file")
-  .pdf

# jax.lax.dynamic_update_slice_in_dim

## Contents

- [`dynamic_update_slice_in_dim()`](#jax.lax.dynamic_update_slice_in_dim)

# jax.lax.dynamic_update_slice_in_dim[\#](#jax-lax-dynamic-update-slice-in-dim "Link to this heading")

jax.lax.dynamic_update_slice_in_dim(*operand*, *update*, *start_index*, *axis*, *\**, *allow_negative_indices=True*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/slicing.py#L1204-L1273)[\#](#jax.lax.dynamic_update_slice_in_dim "Link to this definition")  
Convenience wrapper around [`dynamic_update_slice()`](jax.lax.dynamic_update_slice.html#jax.lax.dynamic_update_slice "jax.lax.dynamic_update_slice") to update a slice in a single `axis`.

Parameters:  
- **operand** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* *np.ndarray*) – an array to slice.

- **update** (*ArrayLike*) – an array containing the new values to write onto operand.

- **start_index** (*ArrayLike*) – a single scalar index

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – the axis of the update.

- **allow_negative_indices** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – boolean specifying whether negative indices are allowed. If true, negative indices are taken relative to the end of the array. If false, negative indices are out of bounds and the result is implementation defined.

Returns:  
The updated array

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Examples

    >>> x = jnp.zeros(6)
    >>> y = jnp.ones(3)
    >>> dynamic_update_slice_in_dim(x, y, 2, axis=0)
    Array([0., 0., 1., 1., 1., 0.], dtype=float32)

If the update slice is too large to fit in the array, the start index will be adjusted to make it fit:

    >>> dynamic_update_slice_in_dim(x, y, 3, axis=0)
    Array([0., 0., 0., 1., 1., 1.], dtype=float32)
    >>> dynamic_update_slice_in_dim(x, y, 5, axis=0)
    Array([0., 0., 0., 1., 1., 1.], dtype=float32)

Here is an example of a two-dimensional slice update:

    >>> x = jnp.zeros((4, 4))
    >>> y = jnp.ones((2, 4))
    >>> dynamic_update_slice_in_dim(x, y, 1, axis=0)
    Array([[0., 0., 0., 0.],
           [1., 1., 1., 1.],
           [1., 1., 1., 1.],
           [0., 0., 0., 0.]], dtype=float32)

Note that the shape of the additional axes in `update` need not match the associated dimensions of the `operand`:

    >>> y = jnp.ones((2, 3))
    >>> dynamic_update_slice_in_dim(x, y, 1, axis=0)
    Array([[0., 0., 0., 0.],
           [1., 1., 1., 0.],
           [1., 1., 1., 0.],
           [0., 0., 0., 0.]], dtype=float32)

See also

- [`jax.numpy.ndarray.at`](jax.numpy.ndarray.at.html#jax.numpy.ndarray.at "jax.numpy.ndarray.at")

- [`jax.lax.dynamic_update_slice()`](jax.lax.dynamic_update_slice.html#jax.lax.dynamic_update_slice "jax.lax.dynamic_update_slice")

- [`jax.lax.dynamic_update_index_in_dim()`](jax.lax.dynamic_update_index_in_dim.html#jax.lax.dynamic_update_index_in_dim "jax.lax.dynamic_update_index_in_dim")

- [`jax.lax.dynamic_slice_in_dim()`](jax.lax.dynamic_slice_in_dim.html#jax.lax.dynamic_slice_in_dim "jax.lax.dynamic_slice_in_dim")

[](jax.lax.dynamic_update_slice.html "previous page")

previous

jax.lax.dynamic_update_slice

[](jax.lax.empty.html "next page")

next

jax.lax.empty

Contents

- [`dynamic_update_slice_in_dim()`](#jax.lax.dynamic_update_slice_in_dim)

By The JAX authors

© Copyright 2024, The JAX Authors.\
