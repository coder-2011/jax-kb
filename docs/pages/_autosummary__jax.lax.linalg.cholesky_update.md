- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.linalg.cholesky_update

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.linalg.cholesky_update.rst "Download source file")
-  .pdf

# jax.lax.linalg.cholesky_update

## Contents

- [`cholesky_update()`](#jax.lax.linalg.cholesky_update)

# jax.lax.linalg.cholesky_update[\#](#jax-lax-linalg-cholesky-update "Link to this heading")

jax.lax.linalg.cholesky_update(*r_matrix*, *w_vector*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/linalg.py#L111-L128)[\#](#jax.lax.linalg.cholesky_update "Link to this definition")  
Cholesky rank-1 update.

Given a Cholesky decomposition \\A = R.T \\ R\\ and a vector \\w\\, computes the Cholesky decomposition of \\A + w \\ w.T\\ in \\O(N^2)\\ time.

Parameters:  
- **r_matrix** (*ArrayLike*) – An upper-triangular matrix (R) such that \\A = R^T \\ R\\.

- **w_vector** (*ArrayLike*) – A vector \\w\\ for rank-1 update.

Returns:  
A new upper-triangular matrix \\R\\ defining the Cholesky decomposition of \\A + w \\ w^T\\.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.lax.linalg.cholesky.html "previous page")

previous

jax.lax.linalg.cholesky

[](jax.lax.linalg.eig.html "next page")

next

jax.lax.linalg.eig

Contents

- [`cholesky_update()`](#jax.lax.linalg.cholesky_update)

By The JAX authors

© Copyright 2024, The JAX Authors.\
