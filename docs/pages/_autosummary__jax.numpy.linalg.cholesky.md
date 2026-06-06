- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.linalg.cholesky

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.linalg.cholesky.rst "Download source file")
-  .pdf

# jax.numpy.linalg.cholesky

## Contents

- [`cholesky()`](#jax.numpy.linalg.cholesky)

# jax.numpy.linalg.cholesky[\#](#jax-numpy-linalg-cholesky "Link to this heading")

jax.numpy.linalg.cholesky(*a*, *\**, *upper=False*, *symmetrize_input=True*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/linalg.py#L80-L150)[\#](#jax.numpy.linalg.cholesky "Link to this definition")  
Compute the Cholesky decomposition of a matrix.

JAX implementation of [`numpy.linalg.cholesky()`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.cholesky.html#numpy.linalg.cholesky "(in NumPy v2.4)").

The Cholesky decomposition of a matrix A is:

\\A = U^HU\\

or

\\A = LL^H\\

where U is an upper-triangular matrix and L is a lower-triangular matrix, and \\X^H\\ is the Hermitian transpose of X.

Parameters:  
- **a** (*ArrayLike*) – input array, representing a (batched) positive-definite hermitian matrix. Must have shape `(...,`` ``N,`` ``N)`.

- **upper** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – if True, compute the upper Cholesky decomposition U. if False (default), compute the lower Cholesky decomposition L.

- **symmetrize_input** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – if True (default) then input is symmetrized, which leads to better behavior under automatic differentiation. Note that when this is set to True, both the upper and lower triangles of the input will be used in computing the decomposition.

Returns:  
array of shape `(...,`` ``N,`` ``N)` representing the Cholesky decomposition of the input. If the input is not Hermitian positive-definite, the result will contain NaN entries.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.scipy.linalg.cholesky()`](jax.scipy.linalg.cholesky.html#jax.scipy.linalg.cholesky "jax.scipy.linalg.cholesky"): SciPy-style Cholesky API

- [`jax.lax.linalg.cholesky()`](jax.lax.linalg.cholesky.html#jax.lax.linalg.cholesky "jax.lax.linalg.cholesky"): XLA-style Cholesky API

Examples

A small real Hermitian positive-definite matrix:

    >>> x = jnp.array([[2., 1.],
    ...                [1., 2.]])

Lower Cholesky factorization:

    >>> jnp.linalg.cholesky(x)
    Array([[1.4142135 , 0.        ],
           [0.70710677, 1.2247449 ]], dtype=float32)

Upper Cholesky factorization:

    >>> jnp.linalg.cholesky(x, upper=True)
    Array([[1.4142135 , 0.70710677],
           [0.        , 1.2247449 ]], dtype=float32)

Reconstructing `x` from its factorization:

    >>> L = jnp.linalg.cholesky(x)
    >>> jnp.allclose(x, L @ L.T)
    Array(True, dtype=bool)

[](jax.numpy.fft.rfftn.html "previous page")

previous

jax.numpy.fft.rfftn

[](jax.numpy.linalg.cond.html "next page")

next

jax.numpy.linalg.cond

Contents

- [`cholesky()`](#jax.numpy.linalg.cholesky)

By The JAX authors

© Copyright 2024, The JAX Authors.\
