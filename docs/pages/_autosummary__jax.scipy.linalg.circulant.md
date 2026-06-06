- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.linalg.circulant

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.linalg.circulant.rst "Download source file")
-  .pdf

# jax.scipy.linalg.circulant

## Contents

- [`circulant()`](#jax.scipy.linalg.circulant)

# jax.scipy.linalg.circulant[\#](#jax-scipy-linalg-circulant "Link to this heading")

jax.scipy.linalg.circulant(*c*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/linalg.py#L2574-L2609)[\#](#jax.scipy.linalg.circulant "Link to this definition")  
Construct a circulant matrix.

JAX implementation of [`scipy.linalg.circulant()`](https://scipy.github.io/devdocs/reference/generated/scipy.linalg.circulant.html#scipy.linalg.circulant "(in SciPy v1.19.0.dev)").

A circulant matrix has cyclically shifted columns: \\A\_{ij} = c\_{(i - j) \bmod n}\\ for \\0 \le i, j \< n\\, where `c` specifies the first column.

Parameters:  
**c** (*ArrayLike*) – array of shape `(...,`` ``N)` specifying the first column.

Returns:  
A circulant matrix of shape `(...,`` ``N,`` ``N)`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Examples

    >>> c = jnp.array([1, 2, 3])
    >>> jax.scipy.linalg.circulant(c)
    Array([[1, 3, 2],
           [2, 1, 3],
           [3, 2, 1]], dtype=int32)

For N-dimensional `c`, the result is a batch of circulant matrices:

    >>> c = jnp.array([[1, 2, 3], [4, 5, 6]])
    >>> jax.scipy.linalg.circulant(c)
    Array([[[1, 3, 2],
            [2, 1, 3],
            [3, 2, 1]],

           [[4, 6, 5],
            [5, 4, 6],
            [6, 5, 4]]], dtype=int32)

[](jax.scipy.linalg.cholesky.html "previous page")

previous

jax.scipy.linalg.cholesky

[](jax.scipy.linalg.companion.html "next page")

next

jax.scipy.linalg.companion

Contents

- [`circulant()`](#jax.scipy.linalg.circulant)

By The JAX authors

© Copyright 2024, The JAX Authors.\
