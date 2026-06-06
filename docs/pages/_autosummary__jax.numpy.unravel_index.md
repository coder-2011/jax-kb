- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.unravel_index

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.unravel_index.rst "Download source file")
-  .pdf

# jax.numpy.unravel_index

## Contents

- [`unravel_index()`](#jax.numpy.unravel_index)

# jax.numpy.unravel_index[\#](#jax-numpy-unravel-index "Link to this heading")

jax.numpy.unravel_index(*indices*, *shape*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L2150-L2215)[\#](#jax.numpy.unravel_index "Link to this definition")  
Convert flat indices into multi-dimensional indices.

JAX implementation of [`numpy.unravel_index()`](https://numpy.org/doc/stable/reference/generated/numpy.unravel_index.html#numpy.unravel_index "(in NumPy v2.4)"). The JAX version differs in its treatment of out-of-bound indices: unlike NumPy, negative indices are supported, and out-of-bound indices are clipped to the nearest valid value.

Parameters:  
- **indices** (*ArrayLike*) – integer array of flat indices

- **shape** (*Shape*) – shape of multidimensional array to index into

Returns:  
Tuple of unraveled indices

Return type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), …\]

See also

[`jax.numpy.ravel_multi_index()`](jax.numpy.ravel_multi_index.html#jax.numpy.ravel_multi_index "jax.numpy.ravel_multi_index"): Inverse of this function.

Examples

Start with a 1D array values and indices:

    >>> x = jnp.array([2., 3., 4., 5., 6., 7.])
    >>> indices = jnp.array([1, 3, 5])
    >>> print(x[indices])
    [3. 5. 7.]

Now if `x` is reshaped, `unravel_indices` can be used to convert the flat indices into a tuple of indices that access the same entries:

    >>> shape = (2, 3)
    >>> x_2D = x.reshape(shape)
    >>> indices_2D = jnp.unravel_index(indices, shape)
    >>> indices_2D
    (Array([0, 1, 1], dtype=int32), Array([1, 0, 2], dtype=int32))
    >>> print(x_2D[indices_2D])
    [3. 5. 7.]

The inverse function, `ravel_multi_index`, can be used to obtain the original indices:

    >>> jnp.ravel_multi_index(indices_2D, shape)
    Array([1, 3, 5], dtype=int32)

[](jax.numpy.unpackbits.html "previous page")

previous

jax.numpy.unpackbits

[](jax.numpy.unstack.html "next page")

next

jax.numpy.unstack

Contents

- [`unravel_index()`](#jax.numpy.unravel_index)

By The JAX authors

© Copyright 2024, The JAX Authors.\
