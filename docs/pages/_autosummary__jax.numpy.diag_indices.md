- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.diag_indices

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.diag_indices.rst "Download source file")
-  .pdf

# jax.numpy.diag_indices

## Contents

- [`diag_indices()`](#jax.numpy.diag_indices)

# jax.numpy.diag_indices[\#](#jax-numpy-diag-indices "Link to this heading")

jax.numpy.diag_indices(*n*, *ndim=2*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L7154-L7194)[\#](#jax.numpy.diag_indices "Link to this definition")  
Return indices for accessing the main diagonal of a multidimensional array.

JAX implementation of [`numpy.diag_indices()`](https://numpy.org/doc/stable/reference/generated/numpy.diag_indices.html#numpy.diag_indices "(in NumPy v2.4)").

Parameters:  
- **n** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – int. The size of each dimension of the square array.

- **ndim** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – optional, int, default=2. The number of dimensions of the array.

Returns:  
A tuple of arrays, each of length n, containing the indices to access the main diagonal.

Return type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), …\]

See also

- [`jax.numpy.diag_indices_from()`](jax.numpy.diag_indices_from.html#jax.numpy.diag_indices_from "jax.numpy.diag_indices_from")

- [`jax.numpy.diagonal()`](jax.numpy.diagonal.html#jax.numpy.diagonal "jax.numpy.diagonal")

Examples

    >>> jnp.diag_indices(3)
    (Array([0, 1, 2], dtype=int32), Array([0, 1, 2], dtype=int32))
    >>> jnp.diag_indices(4, ndim=3)
    (Array([0, 1, 2, 3], dtype=int32),
    Array([0, 1, 2, 3], dtype=int32),
    Array([0, 1, 2, 3], dtype=int32))

[](jax.numpy.diag.html "previous page")

previous

jax.numpy.diag

[](jax.numpy.diag_indices_from.html "next page")

next

jax.numpy.diag_indices_from

Contents

- [`diag_indices()`](#jax.numpy.diag_indices)

By The JAX authors

© Copyright 2024, The JAX Authors.\
