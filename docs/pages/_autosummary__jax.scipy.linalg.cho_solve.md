- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.linalg.cho_solve

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.linalg.cho_solve.rst "Download source file")
-  .pdf

# jax.scipy.linalg.cho_solve

## Contents

- [`cho_solve()`](#jax.scipy.linalg.cho_solve)

# jax.scipy.linalg.cho_solve[\#](#jax-scipy-linalg-cho-solve "Link to this heading")

jax.scipy.linalg.cho_solve(*c_and_lower*, *b*, *overwrite_b=False*, *check_finite=True*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/linalg.py#L188-L237)[\#](#jax.scipy.linalg.cho_solve "Link to this definition")  
Solve a linear system using a Cholesky factorization

JAX implementation of [`scipy.linalg.cho_solve()`](https://scipy.github.io/devdocs/reference/generated/scipy.linalg.cho_solve.html#scipy.linalg.cho_solve "(in SciPy v1.19.0.dev)"). Uses the output of [`jax.scipy.linalg.cho_factor()`](jax.scipy.linalg.cho_factor.html#jax.scipy.linalg.cho_factor "jax.scipy.linalg.cho_factor").

Parameters:  
- **c_and_lower** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[ArrayLike,* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")*\]*) – `(c,`` ``lower)`, where `c` is an array of shape `(...,`` ``N,`` ``N)` representing the lower or upper cholesky decomposition of the matrix, and `lower` is a boolean specifying whether this is the lower or upper decomposition.

- **b** (*ArrayLike*) – right-hand-side of linear system. Array of shape `(N,)` (for a 1-dimensional right-hand-side) or `(...,`` ``N,`` ``M)` (for a batched 2-dimensional right-hand-side).

- **overwrite_a** – unused by JAX

- **check_finite** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – unused by JAX

- **overwrite_b** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

Returns:  
Array representing the solution of the linear system. The result has shape `(...,`` ``N)` if `b` is of shape `(N,)`, and has shape `(...,`` ``N,`` ``M)` otherwise.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.scipy.linalg.cholesky()`](jax.scipy.linalg.cholesky.html#jax.scipy.linalg.cholesky "jax.scipy.linalg.cholesky")

- [`jax.scipy.linalg.cho_factor()`](jax.scipy.linalg.cho_factor.html#jax.scipy.linalg.cho_factor "jax.scipy.linalg.cho_factor")

Examples

A small real Hermitian positive-definite matrix:

    >>> x = jnp.array([[2., 1.],
    ...                [1., 2.]])

Compute the cholesky factorization via [`cho_factor()`](jax.scipy.linalg.cho_factor.html#jax.scipy.linalg.cho_factor "jax.scipy.linalg.cho_factor"), and use it to solve a linear equation via [`cho_solve()`](#jax.scipy.linalg.cho_solve "jax.scipy.linalg.cho_solve").

    >>> b = jnp.array([3., 4.])
    >>> cfac = jax.scipy.linalg.cho_factor(x)
    >>> y = jax.scipy.linalg.cho_solve(cfac, b)
    >>> y
    Array([0.6666666, 1.6666666], dtype=float32)

Check that the result is consistent:

    >>> jnp.allclose(x @ y, b)
    Array(True, dtype=bool)

[](jax.scipy.linalg.cho_factor.html "previous page")

previous

jax.scipy.linalg.cho_factor

[](jax.scipy.linalg.cholesky.html "next page")

next

jax.scipy.linalg.cholesky

Contents

- [`cho_solve()`](#jax.scipy.linalg.cho_solve)

By The JAX authors

© Copyright 2024, The JAX Authors.\
