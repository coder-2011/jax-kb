- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.dynamic_update_slice

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.dynamic_update_slice.rst "Download source file")
-  .pdf

# jax.lax.dynamic_update_slice

## Contents

- [`dynamic_update_slice()`](#jax.lax.dynamic_update_slice)

# jax.lax.dynamic_update_slice[\#](#jax-lax-dynamic-update-slice "Link to this heading")

jax.lax.dynamic_update_slice(*operand*, *update*, *start_indices*, *\**, *allow_negative_indices=True*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/slicing.py#L181-L241)[\#](#jax.lax.dynamic_update_slice "Link to this definition")  
Wraps XLA’s [DynamicUpdateSlice](https://www.openxla.org/xla/operation_semantics#dynamicupdateslice) operator.

Parameters:  
- **operand** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* *np.ndarray*) – an array to slice.

- **update** (*ArrayLike*) – an array containing the new values to write onto operand.

- **start_indices** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* *Sequence\[ArrayLike\]*) – a list of scalar indices, one per dimension.

- **allow_negative_indices** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* *Sequence\[*[*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")*\]*) – a bool or sequence of bools, one per dimension; if a bool is passed, it applies to all dimensions. For each dimension, if true, negative indices are permitted and are are interpreted relative to the end of the array. If false, negative indices are treated as if they were out of bounds and the result is implementation defined, typically clamped to the first index.

Returns:  
An array containing the slice.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Examples

Here is an example of updating a one-dimensional slice update:

    >>> x = jnp.zeros(6)
    >>> y = jnp.ones(3)
    >>> dynamic_update_slice(x, y, (2,))
    Array([0., 0., 1., 1., 1., 0.], dtype=float32)

If the update slice is too large to fit in the array, the start index will be adjusted to make it fit

    >>> dynamic_update_slice(x, y, (3,))
    Array([0., 0., 0., 1., 1., 1.], dtype=float32)
    >>> dynamic_update_slice(x, y, (5,))
    Array([0., 0., 0., 1., 1., 1.], dtype=float32)

Here is an example of a two-dimensional slice update:

    >>> x = jnp.zeros((4, 4))
    >>> y = jnp.ones((2, 2))
    >>> dynamic_update_slice(x, y, (1, 2))
    Array([[0., 0., 0., 0.],
           [0., 0., 1., 1.],
           [0., 0., 1., 1.],
           [0., 0., 0., 0.]], dtype=float32)

See also

- [`jax.numpy.ndarray.at`](jax.numpy.ndarray.at.html#jax.numpy.ndarray.at "jax.numpy.ndarray.at")

- `lax.dynamic_update_index_in_dim`

- `lax.dynamic_update_slice_in_dim`

[](jax.lax.dynamic_update_index_in_dim.html "previous page")

previous

jax.lax.dynamic_update_index_in_dim

[](jax.lax.dynamic_update_slice_in_dim.html "next page")

next

jax.lax.dynamic_update_slice_in_dim

Contents

- [`dynamic_update_slice()`](#jax.lax.dynamic_update_slice)

By The JAX authors

© Copyright 2024, The JAX Authors.\
