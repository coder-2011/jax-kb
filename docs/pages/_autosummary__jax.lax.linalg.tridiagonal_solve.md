- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.linalg.tridiagonal_solve

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.linalg.tridiagonal_solve.rst "Download source file")
-  .pdf

# jax.lax.linalg.tridiagonal_solve

## Contents

- [`tridiagonal_solve()`](#jax.lax.linalg.tridiagonal_solve)

# jax.lax.linalg.tridiagonal_solve[\#](#jax-lax-linalg-tridiagonal-solve "Link to this heading")

jax.lax.linalg.tridiagonal_solve(*dl*, *d*, *du*, *b*, *\**, *perturb_singular=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/linalg.py#L700-L738)[\#](#jax.lax.linalg.tridiagonal_solve "Link to this definition")  
Computes the solution of a tridiagonal linear system.

This function computes the solution of a tridiagonal linear system:

\\A \\ X = B\\

Parameters:  
- **dl** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – A batch of vectors with shape `[...,`` ``m]`. The lower diagonal of A: `dl[i]`` ``:=`` ``A[i,`` ``i-1]` for i in `[0,m)`. Note that `dl[0]`` ``=`` ``0`.

- **d** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – A batch of vectors with shape `[...,`` ``m]`. The middle diagonal of A: `d[i]``  ``:=`` ``A[i,`` ``i]` for i in `[0,m)`.

- **du** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – A batch of vectors with shape `[...,`` ``m]`. The upper diagonal of A: `du[i]`` ``:=`` ``A[i,`` ``i+1]` for i in `[0,m)`. Note that `dl[m`` ``-`` ``1]`` ``=`` ``0`.

- **b** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – Right hand side matrix.

- **perturb_singular** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – Whether to perturb singular matrices to return a finite result. `False` by default. If `True`, solutions to systems involving a singular matrix will be computed by perturbing near-zero pivots in the partially pivoted LU decomposition. Specifically, tiny pivots are perturbed by an amount of order `eps`` ``*`` ``max_{ij}`` ``|U(i,j)|` to avoid overflow. Here `U` is the upper triangular part of the LU decomposition, and `eps` is the machine precision. This is useful for solving numerically singular systems when computing eigenvectors by inverse iteration. Only implemented on CPU and GPU at the moment.

Returns:  
Solution `X` of tridiagonal system.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.lax.linalg.tridiagonal.html "previous page")

previous

jax.lax.linalg.tridiagonal

[](../jax.random.html "next page")

next

`jax.random` module

Contents

- [`tridiagonal_solve()`](#jax.lax.linalg.tridiagonal_solve)

By The JAX authors

© Copyright 2024, The JAX Authors.\
