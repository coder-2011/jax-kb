- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.diag_indices_from

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.diag_indices_from.rst "Download source file")
-  .pdf

# jax.numpy.diag_indices_from

## Contents

- [`diag_indices_from()`](#jax.numpy.diag_indices_from)

# jax.numpy.diag_indices_from[\#](#jax-numpy-diag-indices-from "Link to this heading")

jax.numpy.diag_indices_from(*arr*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L7196-L7237)[\#](#jax.numpy.diag_indices_from "Link to this definition")  
Return indices for accessing the main diagonal of a given array.

JAX implementation of [`numpy.diag_indices_from()`](https://numpy.org/doc/stable/reference/generated/numpy.diag_indices_from.html#numpy.diag_indices_from "(in NumPy v2.4)").

Parameters:  
**arr** (*ArrayLike*) – Input array. Must be at least 2-dimensional and have equal length along all dimensions.

Returns:  
A tuple of arrays containing the indices to access the main diagonal of the input array.

Return type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), …\]

See also

- [`jax.numpy.diag_indices()`](jax.numpy.diag_indices.html#jax.numpy.diag_indices "jax.numpy.diag_indices")

- [`jax.numpy.diagonal()`](jax.numpy.diagonal.html#jax.numpy.diagonal "jax.numpy.diagonal")

Examples

    >>> arr = jnp.array([[1, 2, 3],
    ...                  [4, 5, 6],
    ...                  [7, 8, 9]])
    >>> jnp.diag_indices_from(arr)
    (Array([0, 1, 2], dtype=int32), Array([0, 1, 2], dtype=int32))
    >>> arr = jnp.array([[[1, 2], [3, 4]],
    ...                  [[5, 6], [7, 8]]])
    >>> jnp.diag_indices_from(arr)
    (Array([0, 1], dtype=int32),
    Array([0, 1], dtype=int32),
    Array([0, 1], dtype=int32))

[](jax.numpy.diag_indices.html "previous page")

previous

jax.numpy.diag_indices

[](jax.numpy.diagflat.html "next page")

next

jax.numpy.diagflat

Contents

- [`diag_indices_from()`](#jax.numpy.diag_indices_from)

By The JAX authors

© Copyright 2024, The JAX Authors.\
