- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.linalg.cholesky

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.linalg.cholesky.rst "Download source file")
-  .pdf

# jax.scipy.linalg.cholesky

## Contents

- [`cholesky()`](#jax.scipy.linalg.cholesky)

# jax.scipy.linalg.cholesky[\#](#jax-scipy-linalg-cholesky "Link to this heading")

jax.scipy.linalg.cholesky(*a*, *lower=False*, *overwrite_a=False*, *check_finite=True*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/linalg.py#L56-L114)[\#](#jax.scipy.linalg.cholesky "Link to this definition")  
Compute the Cholesky decomposition of a matrix.

JAX implementation of [`scipy.linalg.cholesky()`](https://scipy.github.io/devdocs/reference/generated/scipy.linalg.cholesky.html#scipy.linalg.cholesky "(in SciPy v1.19.0.dev)").

The Cholesky decomposition of a matrix A is:

\\A = U^HU = LL^H\\

where U is an upper-triangular matrix and L is a lower-triangular matrix.

Parameters:  
- **a** (*ArrayLike*) – input array, representing a (batched) positive-definite hermitian matrix. Must have shape `(...,`` ``N,`` ``N)`.

- **lower** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – if True, compute the lower Cholesky decomposition L. if False (default), compute the upper Cholesky decomposition U.

- **overwrite_a** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – unused by JAX

- **check_finite** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – unused by JAX

Returns:  
array of shape `(...,`` ``N,`` ``N)` representing the cholesky decomposition of the input.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.linalg.cholesky()`](jax.numpy.linalg.cholesky.html#jax.numpy.linalg.cholesky "jax.numpy.linalg.cholesky"): NumPy-stype Cholesky API

- [`jax.lax.linalg.cholesky()`](jax.lax.linalg.cholesky.html#jax.lax.linalg.cholesky "jax.lax.linalg.cholesky"): XLA-style Cholesky API

- [`jax.scipy.linalg.cho_factor()`](jax.scipy.linalg.cho_factor.html#jax.scipy.linalg.cho_factor "jax.scipy.linalg.cho_factor")

- [`jax.scipy.linalg.cho_solve()`](jax.scipy.linalg.cho_solve.html#jax.scipy.linalg.cho_solve "jax.scipy.linalg.cho_solve")

Examples

A small real Hermitian positive-definite matrix:

    >>> x = jnp.array([[2., 1.],
    ...                [1., 2.]])

Upper Cholesky factorization:

    >>> jax.scipy.linalg.cholesky(x)
    Array([[1.4142135 , 0.70710677],
           [0.        , 1.2247449 ]], dtype=float32)

Lower Cholesky factorization:

    >>> jax.scipy.linalg.cholesky(x, lower=True)
    Array([[1.4142135 , 0.        ],
           [0.70710677, 1.2247449 ]], dtype=float32)

Reconstructing `x` from its factorization:

    >>> L = jax.scipy.linalg.cholesky(x, lower=True)
    >>> jnp.allclose(x, L @ L.T)
    Array(True, dtype=bool)

[](jax.scipy.linalg.cho_solve.html "previous page")

previous

jax.scipy.linalg.cho_solve

[](jax.scipy.linalg.circulant.html "next page")

next

jax.scipy.linalg.circulant

Contents

- [`cholesky()`](#jax.scipy.linalg.cholesky)

By The JAX authors

© Copyright 2024, The JAX Authors.\
