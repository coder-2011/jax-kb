- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.linalg.tridiagonal

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.linalg.tridiagonal.rst "Download source file")
-  .pdf

# jax.lax.linalg.tridiagonal

## Contents

- [`tridiagonal()`](#jax.lax.linalg.tridiagonal)

# jax.lax.linalg.tridiagonal[\#](#jax-lax-linalg-tridiagonal "Link to this heading")

jax.lax.linalg.tridiagonal(*a*, *\**, *lower=True*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/linalg.py#L672-L698)[\#](#jax.lax.linalg.tridiagonal "Link to this definition")  
Reduces a symmetric/Hermitian matrix to tridiagonal form.

Currently implemented on CPU and GPU only.

Parameters:  
- **a** (*ArrayLike*) – A floating point or complex matrix or batch of matrices.

- **lower** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – Describes which triangle of the input matrices to use. The other triangle is ignored and not accessed.

Returns:  
A `(a,`` ``d,`` ``e,`` ``taus)` tuple. If `lower=True`, the diagonal and first subdiagonal of matrix (or batch of matrices) `a` contain the tridiagonal representation, and elements below the first subdiagonal contain the elementary Householder reflectors, where additionally `d` contains the diagonal of the matrix and `e` contains the first subdiagonal. If `lower=False` the diagonal and first superdiagonal of the matrix contains the tridiagonal representation, and elements above the first superdiagonal contain the elementary Householder reflectors, where additionally `d` contains the diagonal of the matrix and `e` contains the first superdiagonal. `taus` contains the scalar factors of the elementary Householder reflectors.

Return type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\]

[](jax.lax.linalg.triangular_solve.html "previous page")

previous

jax.lax.linalg.triangular_solve

[](jax.lax.linalg.tridiagonal_solve.html "next page")

next

jax.lax.linalg.tridiagonal_solve

Contents

- [`tridiagonal()`](#jax.lax.linalg.tridiagonal)

By The JAX authors

© Copyright 2024, The JAX Authors.\
