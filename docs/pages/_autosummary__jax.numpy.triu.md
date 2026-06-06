- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.triu

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.triu.rst "Download source file")
-  .pdf

# jax.numpy.triu

## Contents

- [`triu()`](#jax.numpy.triu)

# jax.numpy.triu[\#](#jax-numpy-triu "Link to this heading")

jax.numpy.triu(*m*, *k=0*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L6635-L6702)[\#](#jax.numpy.triu "Link to this definition")  
Return upper triangle of an array.

JAX implementation of [`numpy.triu()`](https://numpy.org/doc/stable/reference/generated/numpy.triu.html#numpy.triu "(in NumPy v2.4)")

Parameters:  
- **m** (*ArrayLike*) – input array. Must have `m.ndim`` ``>=`` ``2`.

- **k** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – optional, int, default=0. Specifies the sub-diagonal below which the elements of the array are set to zero. `k=0` refers to main diagonal, `k<0` refers to sub-diagonal below the main diagonal and `k>0` refers to sub-diagonal above the main diagonal.

Returns:  
An array with same shape as input containing the upper triangle of the given array with elements below the sub-diagonal specified by `k` are set to zero.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.tril()`](jax.numpy.tril.html#jax.numpy.tril "jax.numpy.tril"): Returns a lower triangle of an array.

- [`jax.numpy.tri()`](jax.numpy.tri.html#jax.numpy.tri "jax.numpy.tri"): Returns an array with ones on and below the diagonal and zeros elsewhere.

Examples

    >>> x = jnp.array([[1, 2, 3],
    ...                [4, 5, 6],
    ...                [7, 8, 9],
    ...                [10, 11, 12]])
    >>> jnp.triu(x)
    Array([[1, 2, 3],
           [0, 5, 6],
           [0, 0, 9],
           [0, 0, 0]], dtype=int32)
    >>> jnp.triu(x, k=1)
    Array([[0, 2, 3],
           [0, 0, 6],
           [0, 0, 0],
           [0, 0, 0]], dtype=int32)
    >>> jnp.triu(x, k=-1)
    Array([[ 1,  2,  3],
           [ 4,  5,  6],
           [ 0,  8,  9],
           [ 0,  0, 12]], dtype=int32)

When `m.ndim`` ``>`` ``2`, `jnp.triu` operates batch-wise on the trailing axes.

    >>> x1 = jnp.array([[[1, 2],
    ...                  [3, 4]],
    ...                 [[5, 6],
    ...                  [7, 8]]])
    >>> jnp.triu(x1)
    Array([[[1, 2],
            [0, 4]],

           [[5, 6],
            [0, 8]]], dtype=int32)

[](jax.numpy.trim_zeros.html "previous page")

previous

jax.numpy.trim_zeros

[](jax.numpy.triu_indices.html "next page")

next

jax.numpy.triu_indices

Contents

- [`triu()`](#jax.numpy.triu)

By The JAX authors

© Copyright 2024, The JAX Authors.\
