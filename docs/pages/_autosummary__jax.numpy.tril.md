- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.tril

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.tril.rst "Download source file")
-  .pdf

# jax.numpy.tril

## Contents

- [`tril()`](#jax.numpy.tril)

# jax.numpy.tril[\#](#jax-numpy-tril "Link to this heading")

jax.numpy.tril(*m*, *k=0*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L6570-L6633)[\#](#jax.numpy.tril "Link to this definition")  
Return lower triangle of an array.

JAX implementation of [`numpy.tril()`](https://numpy.org/doc/stable/reference/generated/numpy.tril.html#numpy.tril "(in NumPy v2.4)")

Parameters:  
- **m** (*ArrayLike*) – input array. Must have `m.ndim`` ``>=`` ``2`.

- **k** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – k: optional, int, default=0. Specifies the sub-diagonal above which the elements of the array are set to zero. `k=0` refers to main diagonal, `k<0` refers to sub-diagonal below the main diagonal and `k>0` refers to sub-diagonal above the main diagonal.

Returns:  
An array with same shape as input containing the lower triangle of the given array with elements above the sub-diagonal specified by `k` are set to zero.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.triu()`](jax.numpy.triu.html#jax.numpy.triu "jax.numpy.triu"): Returns an upper triangle of an array.

- [`jax.numpy.tri()`](jax.numpy.tri.html#jax.numpy.tri "jax.numpy.tri"): Returns an array with ones on and below the diagonal and zeros elsewhere.

Examples

    >>> x = jnp.array([[1, 2, 3, 4],
    ...                [5, 6, 7, 8],
    ...                [9, 10, 11, 12]])
    >>> jnp.tril(x)
    Array([[ 1,  0,  0,  0],
           [ 5,  6,  0,  0],
           [ 9, 10, 11,  0]], dtype=int32)
    >>> jnp.tril(x, k=1)
    Array([[ 1,  2,  0,  0],
           [ 5,  6,  7,  0],
           [ 9, 10, 11, 12]], dtype=int32)
    >>> jnp.tril(x, k=-1)
    Array([[ 0,  0,  0,  0],
           [ 5,  0,  0,  0],
           [ 9, 10,  0,  0]], dtype=int32)

When `m.ndim`` ``>`` ``2`, `jnp.tril` operates batch-wise on the trailing axes.

    >>> x1 = jnp.array([[[1, 2],
    ...                  [3, 4]],
    ...                 [[5, 6],
    ...                  [7, 8]]])
    >>> jnp.tril(x1)
    Array([[[1, 0],
            [3, 4]],

           [[5, 0],
            [7, 8]]], dtype=int32)

[](jax.numpy.tri.html "previous page")

previous

jax.numpy.tri

[](jax.numpy.tril_indices.html "next page")

next

jax.numpy.tril_indices

Contents

- [`tril()`](#jax.numpy.tril)

By The JAX authors

© Copyright 2024, The JAX Authors.\
