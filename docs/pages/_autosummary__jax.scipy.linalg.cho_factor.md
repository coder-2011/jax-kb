- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.linalg.cho_factor

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.linalg.cho_factor.rst "Download source file")
-  .pdf

# jax.scipy.linalg.cho_factor

## Contents

- [`cho_factor()`](#jax.scipy.linalg.cho_factor)

# jax.scipy.linalg.cho_factor[\#](#jax-scipy-linalg-cho-factor "Link to this heading")

jax.scipy.linalg.cho_factor(*a*, *lower=False*, *overwrite_a=False*, *check_finite=True*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/linalg.py#L116-L162)[\#](#jax.scipy.linalg.cho_factor "Link to this definition")  
Factorization for Cholesky-based linear solves

JAX implementation of [`scipy.linalg.cho_factor()`](https://scipy.github.io/devdocs/reference/generated/scipy.linalg.cho_factor.html#scipy.linalg.cho_factor "(in SciPy v1.19.0.dev)"). This function returns a result suitable for use with [`jax.scipy.linalg.cho_solve()`](jax.scipy.linalg.cho_solve.html#jax.scipy.linalg.cho_solve "jax.scipy.linalg.cho_solve"). For direct Cholesky decompositions, prefer [`jax.scipy.linalg.cholesky()`](jax.scipy.linalg.cholesky.html#jax.scipy.linalg.cholesky "jax.scipy.linalg.cholesky").

Parameters:  
- **a** (*ArrayLike*) – input array, representing a (batched) positive-definite hermitian matrix. Must have shape `(...,`` ``N,`` ``N)`.

- **lower** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – if True, compute the lower triangular Cholesky decomposition (default: False).

- **overwrite_a** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – unused by JAX

- **check_finite** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – unused by JAX

Returns:  
`c` is an array of shape `(...,`` ``N,`` ``N)` representing the lower or upper cholesky decomposition of the input; `lower` is a boolean specifying whether this is the lower or upper decomposition.

Return type:  
`(c,`` ``lower)`

See also

- [`jax.scipy.linalg.cholesky()`](jax.scipy.linalg.cholesky.html#jax.scipy.linalg.cholesky "jax.scipy.linalg.cholesky")

- [`jax.scipy.linalg.cho_solve()`](jax.scipy.linalg.cho_solve.html#jax.scipy.linalg.cho_solve "jax.scipy.linalg.cho_solve")

Examples

A small real Hermitian positive-definite matrix:

    >>> x = jnp.array([[2., 1.],
    ...                [1., 2.]])

Compute the cholesky factorization via [`cho_factor()`](#jax.scipy.linalg.cho_factor "jax.scipy.linalg.cho_factor"), and use it to solve a linear equation via [`cho_solve()`](jax.scipy.linalg.cho_solve.html#jax.scipy.linalg.cho_solve "jax.scipy.linalg.cho_solve").

    >>> b = jnp.array([3., 4.])
    >>> cfac = jax.scipy.linalg.cho_factor(x)
    >>> y = jax.scipy.linalg.cho_solve(cfac, b)
    >>> y
    Array([0.6666666, 1.6666666], dtype=float32)

Check that the result is consistent:

    >>> jnp.allclose(x @ y, b)
    Array(True, dtype=bool)

[](jax.scipy.linalg.block_diag.html "previous page")

previous

jax.scipy.linalg.block_diag

[](jax.scipy.linalg.cho_solve.html "next page")

next

jax.scipy.linalg.cho_solve

Contents

- [`cho_factor()`](#jax.scipy.linalg.cho_factor)

By The JAX authors

© Copyright 2024, The JAX Authors.\
