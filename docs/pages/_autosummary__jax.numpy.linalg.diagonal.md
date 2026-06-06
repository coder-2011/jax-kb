- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.linalg.diagonal

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.linalg.diagonal.rst "Download source file")
-  .pdf

# jax.numpy.linalg.diagonal

## Contents

- [`diagonal()`](#jax.numpy.linalg.diagonal)

# jax.numpy.linalg.diagonal[\#](#jax-numpy-linalg-diagonal "Link to this heading")

jax.numpy.linalg.diagonal(*x*, */*, *\**, *offset=0*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/linalg.py#L1980-L2019)[\#](#jax.numpy.linalg.diagonal "Link to this definition")  
Extract the diagonal of an matrix or stack of matrices.

JAX implementation of [`numpy.linalg.diagonal()`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.diagonal.html#numpy.linalg.diagonal "(in NumPy v2.4)").

Parameters:  
- **x** (*ArrayLike*) – array of shape `(...,`` ``M,`` ``N)` from which the diagonal will be extracted.

- **offset** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – positive or negative offset from the main diagonal.

Returns:  
Array of shape `(...,`` ``K)` where `K` is the length of the specified diagonal.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.diagonal()`](jax.numpy.diagonal.html#jax.numpy.diagonal "jax.numpy.diagonal"): more general functionality for extracting diagonals.

- [`jax.numpy.diag()`](jax.numpy.diag.html#jax.numpy.diag "jax.numpy.diag"): create a diagonal matrix from values.

Examples

Diagonals of a single matrix:

    >>> x = jnp.array([[1,  2,  3,  4],
    ...                [5,  6,  7,  8],
    ...                [9, 10, 11, 12]])
    >>> jnp.linalg.diagonal(x)
    Array([ 1,  6, 11], dtype=int32)
    >>> jnp.linalg.diagonal(x, offset=1)
    Array([ 2,  7, 12], dtype=int32)
    >>> jnp.linalg.diagonal(x, offset=-1)
    Array([ 5, 10], dtype=int32)

Batched diagonals:

    >>> x = jnp.arange(24).reshape(2, 3, 4)
    >>> jnp.linalg.diagonal(x)
    Array([[ 0,  5, 10],
           [12, 17, 22]], dtype=int32)

[](jax.numpy.linalg.det.html "previous page")

previous

jax.numpy.linalg.det

[](jax.numpy.linalg.eig.html "next page")

next

jax.numpy.linalg.eig

Contents

- [`diagonal()`](#jax.numpy.linalg.diagonal)

By The JAX authors

© Copyright 2024, The JAX Authors.\
