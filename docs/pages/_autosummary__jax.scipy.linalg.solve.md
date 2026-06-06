- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.linalg.solve

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.linalg.solve.rst "Download source file")
-  .pdf

# jax.scipy.linalg.solve

## Contents

- [`solve()`](#jax.scipy.linalg.solve)

# jax.scipy.linalg.solve[\#](#jax-scipy-linalg-solve "Link to this heading")

jax.scipy.linalg.solve(*a*, *b*, *lower=False*, *overwrite_a=False*, *overwrite_b=False*, *debug=False*, *check_finite=True*, *assume_a='gen'*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/linalg.py#L1206-L1268)[\#](#jax.scipy.linalg.solve "Link to this definition")  
Solve a linear system of equations.

JAX implementation of [`scipy.linalg.solve()`](https://scipy.github.io/devdocs/reference/generated/scipy.linalg.solve.html#scipy.linalg.solve "(in SciPy v1.19.0.dev)").

This solves a (batched) linear system of equations `a`` ``@`` ``x`` ``=`` ``b` for `x` given `a` and `b`.

If `a` is singular, this will return `nan` or `inf` values.

Parameters:  
- **a** (*ArrayLike*) – array of shape `(...,`` ``N,`` ``N)`.

- **b** (*ArrayLike*) – array of shape `(...,`` ``N)` or `(...,`` ``N,`` ``M)`

- **lower** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – Referenced only if `assume_a`` ``!=`` ``'gen'`. If True, only use the lower triangle of the input, If False (default), only use the upper triangle.

- **assume_a** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) –

  specify what properties of `a` can be assumed. Options are:

  - `"gen"`: generic matrix (default)

  - `"sym"`: symmetric matrix

  - `"her"`: hermitian matrix

  - `"pos"`: positive-definite matrix

- **overwrite_a** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – unused by JAX

- **overwrite_b** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – unused by JAX

- **debug** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – unused by JAX

- **check_finite** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – unused by JAX

Returns:  
An array of the same shape as `b` containing the solution to the linear system if `a` is non-singular. If `a` is singular, the result contains `nan` or `inf` values.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.scipy.linalg.lu_solve()`](jax.scipy.linalg.lu_solve.html#jax.scipy.linalg.lu_solve "jax.scipy.linalg.lu_solve"): Solve via LU factorization.

- [`jax.scipy.linalg.cho_solve()`](jax.scipy.linalg.cho_solve.html#jax.scipy.linalg.cho_solve "jax.scipy.linalg.cho_solve"): Solve via Cholesky factorization.

- [`jax.scipy.linalg.solve_triangular()`](jax.scipy.linalg.solve_triangular.html#jax.scipy.linalg.solve_triangular "jax.scipy.linalg.solve_triangular"): Solve a triangular system.

- [`jax.numpy.linalg.solve()`](jax.numpy.linalg.solve.html#jax.numpy.linalg.solve "jax.numpy.linalg.solve"): NumPy-style API for solving linear systems.

- [`jax.lax.custom_linear_solve()`](jax.lax.custom_linear_solve.html#jax.lax.custom_linear_solve "jax.lax.custom_linear_solve"): matrix-free linear solver.

Examples

A simple 3x3 linear system:

    >>> A = jnp.array([[1., 2., 3.],
    ...                [2., 4., 2.],
    ...                [3., 2., 1.]])
    >>> b = jnp.array([14., 16., 10.])
    >>> x = jax.scipy.linalg.solve(A, b)
    >>> x
    Array([1., 2., 3.], dtype=float32)

Confirming that the result solves the system:

    >>> jnp.allclose(A @ x, b)
    Array(True, dtype=bool)

[](jax.scipy.linalg.schur.html "previous page")

previous

jax.scipy.linalg.schur

[](jax.scipy.linalg.solve_sylvester.html "next page")

next

jax.scipy.linalg.solve_sylvester

Contents

- [`solve()`](#jax.scipy.linalg.solve)

By The JAX authors

© Copyright 2024, The JAX Authors.\
