- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.linalg.solve_triangular

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.linalg.solve_triangular.rst "Download source file")
-  .pdf

# jax.scipy.linalg.solve_triangular

## Contents

- [`solve_triangular()`](#jax.scipy.linalg.solve_triangular)

# jax.scipy.linalg.solve_triangular[\#](#jax-scipy-linalg-solve-triangular "Link to this heading")

jax.scipy.linalg.solve_triangular(*a*, *b*, *trans=0*, *lower=False*, *unit_diagonal=False*, *overwrite_b=False*, *debug=None*, *check_finite=True*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/linalg.py#L1303-L1369)[\#](#jax.scipy.linalg.solve_triangular "Link to this definition")  
Solve a triangular linear system of equations

JAX implementation of [`scipy.linalg.solve_triangular()`](https://scipy.github.io/devdocs/reference/generated/scipy.linalg.solve_triangular.html#scipy.linalg.solve_triangular "(in SciPy v1.19.0.dev)").

This solves a (batched) linear system of equations `a`` ``@`` ``x`` ``=`` ``b` for `x` given a triangular matrix `a` and a vector or matrix `b`.

Parameters:  
- **a** (*ArrayLike*) – array of shape `(...,`` ``N,`` ``N)`. Only part of the array will be accessed, depending on the `lower` and `unit_diagonal` arguments.

- **b** (*ArrayLike*) – array of shape `(N,)` (for a 1-dimensional right-hand-side) or `(...,`` ``N,`` ``M)` (for a batched 2-dimensional right-hand-side).

- **lower** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – If True, only use the lower triangle of the input, If False (default), only use the upper triangle.

- **unit_diagonal** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – If True, ignore diagonal elements of `a` and assume they are `1` (default: False).

- **trans** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) –

  specify what properties of `a` can be assumed. Options are:

  - `0` or `'N'`: solve \\Ax=b\\

  - `1` or `'T'`: solve \\A^Tx=b\\

  - `2` or `'C'`: solve \\A^Hx=b\\

- **overwrite_b** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – unused by JAX

- **debug** (*Any*) – unused by JAX

- **check_finite** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – unused by JAX

Returns:  
An array containing the solution to the linear system. The result has shape `(...,`` ``N)` if `b` is of shape `(N,)`, and has shape `(...,`` ``N,`` ``M)` otherwise.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

[`jax.scipy.linalg.solve()`](jax.scipy.linalg.solve.html#jax.scipy.linalg.solve "jax.scipy.linalg.solve"): Solve a general linear system.

Examples

A simple 3x3 triangular linear system:

    >>> A = jnp.array([[1., 2., 3.],
    ...                [0., 3., 2.],
    ...                [0., 0., 5.]])
    >>> b = jnp.array([10., 8., 5.])
    >>> x = jax.scipy.linalg.solve_triangular(A, b)
    >>> x
    Array([3., 2., 1.], dtype=float32)

Confirming that the result solves the system:

    >>> jnp.allclose(A @ x, b)
    Array(True, dtype=bool)

Computing the transposed problem:

    >>> x = jax.scipy.linalg.solve_triangular(A, b, trans='T')
    >>> x
    Array([10. , -4. , -3.4], dtype=float32)

Confirming that the result solves the system:

    >>> jnp.allclose(A.T @ x, b)
    Array(True, dtype=bool)

[](jax.scipy.linalg.solve_sylvester.html "previous page")

previous

jax.scipy.linalg.solve_sylvester

[](jax.scipy.linalg.sqrtm.html "next page")

next

jax.scipy.linalg.sqrtm

Contents

- [`solve_triangular()`](#jax.scipy.linalg.solve_triangular)

By The JAX authors

© Copyright 2024, The JAX Authors.\
