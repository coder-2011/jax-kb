- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.linalg.symmetric_product

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.linalg.symmetric_product.rst "Download source file")
-  .pdf

# jax.lax.linalg.symmetric_product

## Contents

- [`symmetric_product()`](#jax.lax.linalg.symmetric_product)

# jax.lax.linalg.symmetric_product[\#](#jax-lax-linalg-symmetric-product "Link to this heading")

jax.lax.linalg.symmetric_product(*a_matrix*, *c_matrix*, *\**, *alpha=1.0*, *beta=0.0*, *symmetrize_output=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/linalg.py#L574-L613)[\#](#jax.lax.linalg.symmetric_product "Link to this definition")  
Symmetric product.

Computes the symmetric product

\\\alpha \\ A \\ A^T + \beta \\ C\\

where \\A\\ is a rectangular matrix and \\C\\ is a symmetric matrix.

Parameters:  
- **a_matrix** (*ArrayLike*) – A batch of matrices with shape `[...,`` ``m,`` ``n]`.

- **c_matrix** (*ArrayLike*) – A batch of matrices with shape `[...,`` ``m,`` ``m]`.

- **alpha** ([*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)")) – A scalar.

- **beta** ([*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)")) – A scalar.

- **symmetrize_output** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – If `True`, the upper triangle of the output is replaced with its transpose.

Returns:  
A batch of matrices with shape `[...,`` ``m,`` ``m]` where only the lower triangle is guaranteed to include the correct values on all platforms. If `symmetrize_output` is `True`, the upper triangle is filled with the transpose of the lower triangle, and the whole matrix is valid.

[](jax.lax.linalg.SvdAlgorithm.html "previous page")

previous

jax.lax.linalg.SvdAlgorithm

[](jax.lax.linalg.triangular_solve.html "next page")

next

jax.lax.linalg.triangular_solve

Contents

- [`symmetric_product()`](#jax.lax.linalg.symmetric_product)

By The JAX authors

© Copyright 2024, The JAX Authors.\
