- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.diagflat

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.diagflat.rst "Download source file")
-  .pdf

# jax.numpy.diagflat

## Contents

- [`diagflat()`](#jax.numpy.diagflat)

# jax.numpy.diagflat[\#](#jax-numpy-diagflat "Link to this heading")

jax.numpy.diagflat(*v*, *k=0*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L7379-L7433)[\#](#jax.numpy.diagflat "Link to this definition")  
Return a 2-D array with the flattened input array laid out on the diagonal.

JAX implementation of [`numpy.diagflat()`](https://numpy.org/doc/stable/reference/generated/numpy.diagflat.html#numpy.diagflat "(in NumPy v2.4)").

This differs from np.diagflat for some scalar values of v. JAX always returns a two-dimensional array, whereas NumPy may return a scalar depending on the type of v.

Parameters:  
- **v** (*ArrayLike*) – Input array. Can be N-dimensional but is flattened to 1D.

- **k** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – optional, default=0. Diagonal offset. Positive values place the diagonal above the main diagonal, negative values place it below the main diagonal.

Returns:  
A 2D array with the input elements placed along the diagonal with the specified offset (k). The remaining entries are filled with zeros.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.diag()`](jax.numpy.diag.html#jax.numpy.diag "jax.numpy.diag")

- [`jax.numpy.diagonal()`](jax.numpy.diagonal.html#jax.numpy.diagonal "jax.numpy.diagonal")

Examples

    >>> jnp.diagflat(jnp.array([1, 2, 3]))
    Array([[1, 0, 0],
           [0, 2, 0],
           [0, 0, 3]], dtype=int32)
    >>> jnp.diagflat(jnp.array([1, 2, 3]), k=1)
    Array([[0, 1, 0, 0],
           [0, 0, 2, 0],
           [0, 0, 0, 3],
           [0, 0, 0, 0]], dtype=int32)
    >>> a = jnp.array([[1, 2],
    ...                [3, 4]])
    >>> jnp.diagflat(a)
    Array([[1, 0, 0, 0],
           [0, 2, 0, 0],
           [0, 0, 3, 0],
           [0, 0, 0, 4]], dtype=int32)

[](jax.numpy.diag_indices_from.html "previous page")

previous

jax.numpy.diag_indices_from

[](jax.numpy.diagonal.html "next page")

next

jax.numpy.diagonal

Contents

- [`diagflat()`](#jax.numpy.diagflat)

By The JAX authors

© Copyright 2024, The JAX Authors.\
