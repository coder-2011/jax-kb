- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.tri

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.tri.rst "Download source file")
-  .pdf

# jax.numpy.tri

## Contents

- [`tri()`](#jax.numpy.tri)

# jax.numpy.tri[\#](#jax-numpy-tri "Link to this heading")

jax.numpy.tri(*N*, *M=None*, *k=0*, *dtype=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L6510-L6568)[\#](#jax.numpy.tri "Link to this definition")  
Return an array with ones on and below the diagonal and zeros elsewhere.

JAX implementation of [`numpy.tri()`](https://numpy.org/doc/stable/reference/generated/numpy.tri.html#numpy.tri "(in NumPy v2.4)")

Parameters:  
- **N** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – int. Dimension of the rows of the returned array.

- **M** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – optional, int. Dimension of the columns of the returned array. If not specified, then `M`` ``=`` ``N`.

- **k** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – optional, int, default=0. Specifies the sub-diagonal on and below which the array is filled with ones. `k=0` refers to main diagonal, `k<0` refers to sub-diagonal below the main diagonal and `k>0` refers to sub-diagonal above the main diagonal.

- **dtype** (*DTypeLike* *\|* *None*) – optional, data type of the returned array. The default type is float.

Returns:  
An array of shape `(N,`` ``M)` containing the lower triangle with elements below the sub-diagonal specified by `k` are set to one and zero elsewhere.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.tril()`](jax.numpy.tril.html#jax.numpy.tril "jax.numpy.tril"): Returns a lower triangle of an array.

- [`jax.numpy.triu()`](jax.numpy.triu.html#jax.numpy.triu "jax.numpy.triu"): Returns an upper triangle of an array.

Examples

    >>> jnp.tri(3)
    Array([[1., 0., 0.],
           [1., 1., 0.],
           [1., 1., 1.]], dtype=float32)

When `M` is not equal to `N`:

    >>> jnp.tri(3, 4)
    Array([[1., 0., 0., 0.],
           [1., 1., 0., 0.],
           [1., 1., 1., 0.]], dtype=float32)

when `k>0`:

    >>> jnp.tri(3, k=1)
    Array([[1., 1., 0.],
           [1., 1., 1.],
           [1., 1., 1.]], dtype=float32)

When `k<0`:

    >>> jnp.tri(3, 4, k=-1)
    Array([[0., 0., 0., 0.],
           [1., 0., 0., 0.],
           [1., 1., 0., 0.]], dtype=float32)

[](jax.numpy.transpose.html "previous page")

previous

jax.numpy.transpose

[](jax.numpy.tril.html "next page")

next

jax.numpy.tril

Contents

- [`tri()`](#jax.numpy.tri)

By The JAX authors

© Copyright 2024, The JAX Authors.\
