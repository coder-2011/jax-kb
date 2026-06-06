- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.linalg.triangular_solve

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.linalg.triangular_solve.rst "Download source file")
-  .pdf

# jax.lax.linalg.triangular_solve

## Contents

- [`triangular_solve()`](#jax.lax.linalg.triangular_solve)

# jax.lax.linalg.triangular_solve[\#](#jax-lax-linalg-triangular-solve "Link to this heading")

jax.lax.linalg.triangular_solve(*a*, *b*, *\**, *left_side=False*, *lower=False*, *transpose_a=False*, *conjugate_a=False*, *unit_diagonal=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/linalg.py#L615-L670)[\#](#jax.lax.linalg.triangular_solve "Link to this definition")  
Triangular solve.

Solves either the matrix equation

\\\mathit{op}(A) . X = B\\

if `left_side` is `True` or

\\X . \mathit{op}(A) = B\\

if `left_side` is `False`.

`A` must be a lower or upper triangular square matrix, and where \\\mathit{op}(A)\\ may either transpose \\A\\ if `transpose_a` is `True` and/or take its complex conjugate if `conjugate_a` is `True`.

Parameters:  
- **a** (*ArrayLike*) – A batch of matrices with shape `[...,`` ``m,`` ``m]`.

- **b** (*ArrayLike*) – A batch of matrices with shape `[...,`` ``m,`` ``n]` if `left_side` is `True` or shape `[...,`` ``n,`` ``m]` otherwise.

- **left_side** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – describes which of the two matrix equations to solve; see above.

- **lower** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – describes which triangle of `a` should be used. The other triangle is ignored.

- **transpose_a** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – if `True`, the value of `a` is transposed.

- **conjugate_a** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – if `True`, the complex conjugate of `a` is used in the solve. Has no effect if `a` is real.

- **unit_diagonal** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – if `True`, the diagonal of `a` is assumed to be unit (all 1s) and not accessed.

Returns:  
A batch of matrices the same shape and dtype as `b`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.lax.linalg.symmetric_product.html "previous page")

previous

jax.lax.linalg.symmetric_product

[](jax.lax.linalg.tridiagonal.html "next page")

next

jax.lax.linalg.tridiagonal

Contents

- [`triangular_solve()`](#jax.lax.linalg.triangular_solve)

By The JAX authors

© Copyright 2024, The JAX Authors.\
