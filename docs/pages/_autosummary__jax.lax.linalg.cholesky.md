- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.linalg.cholesky

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.linalg.cholesky.rst "Download source file")
-  .pdf

# jax.lax.linalg.cholesky

## Contents

- [`cholesky()`](#jax.lax.linalg.cholesky)

# jax.lax.linalg.cholesky[\#](#jax-lax-linalg-cholesky "Link to this heading")

jax.lax.linalg.cholesky(*x*, *\**, *symmetrize_input=True*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/linalg.py#L81-L109)[\#](#jax.lax.linalg.cholesky "Link to this definition")  
Cholesky decomposition.

Computes the Cholesky decomposition

\\A = L . L^H\\

of square matrices, \\A\\, such that \\L\\ is lower triangular. The matrices of \\A\\ must be positive-definite and either Hermitian, if complex, or symmetric, if real.

Parameters:  
- **x** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – A batch of square Hermitian (symmetric if real) positive-definite matrices with shape `[...,`` ``n,`` ``n]`.

- **symmetrize_input** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – If `True`, the matrix is symmetrized before Cholesky decomposition by computing \\\frac{1}{2}(x + x^H)\\. If `False`, only the lower triangle of `x` is used; the upper triangle is ignored and not accessed.

Returns:  
The Cholesky decomposition as a matrix with the same dtype as `x` and shape `[...,`` ``n,`` ``n]`. If Cholesky decomposition fails, returns a matrix full of NaNs. The behavior on failure may change in the future.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.lax.with_sharding_constraint.html "previous page")

previous

jax.lax.with_sharding_constraint

[](jax.lax.linalg.cholesky_update.html "next page")

next

jax.lax.linalg.cholesky_update

Contents

- [`cholesky()`](#jax.lax.linalg.cholesky)

By The JAX authors

© Copyright 2024, The JAX Authors.\
