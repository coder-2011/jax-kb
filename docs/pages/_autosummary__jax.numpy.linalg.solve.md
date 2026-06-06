- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.linalg.solve

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.linalg.solve.rst "Download source file")
-  .pdf

# jax.numpy.linalg.solve

## Contents

- [`solve()`](#jax.numpy.linalg.solve)

# jax.numpy.linalg.solve[\#](#jax-numpy-linalg-solve "Link to this heading")

jax.numpy.linalg.solve(*a*, *b*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/linalg.py#L1379-L1440)[\#](#jax.numpy.linalg.solve "Link to this definition")  
Solve a linear system of equations.

JAX implementation of [`numpy.linalg.solve()`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html#numpy.linalg.solve "(in NumPy v2.4)").

This solves a (batched) linear system of equations `a`` ``@`` ``x`` ``=`` ``b` for `x` given `a` and `b`.

If `a` is singular, this will return `nan` or `inf` values.

Parameters:  
- **a** (*ArrayLike*) – array of shape `(...,`` ``N,`` ``N)`.

- **b** (*ArrayLike*) – array of shape `(N,)` (for 1-dimensional right-hand-side) or `(...,`` ``N,`` ``M)` (for batched 2-dimensional right-hand-side).

Returns:  
An array containing the result of the linear solve if `a` is non-singular. The result has shape `(...,`` ``N)` if `b` is of shape `(N,)`, and has shape `(...,`` ``N,`` ``M)` otherwise. If `a` is singular, the result contains `nan` or `inf` values.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.scipy.linalg.solve()`](jax.scipy.linalg.solve.html#jax.scipy.linalg.solve "jax.scipy.linalg.solve"): SciPy-style API for solving linear systems.

- [`jax.lax.custom_linear_solve()`](jax.lax.custom_linear_solve.html#jax.lax.custom_linear_solve "jax.lax.custom_linear_solve"): matrix-free linear solver.

Examples

A simple 3x3 linear system:

    >>> A = jnp.array([[1., 2., 3.],
    ...                [2., 4., 2.],
    ...                [3., 2., 1.]])
    >>> b = jnp.array([14., 16., 10.])
    >>> x = jnp.linalg.solve(A, b)
    >>> x
    Array([1., 2., 3.], dtype=float32)

Confirming that the result solves the system:

    >>> jnp.allclose(A @ x, b)
    Array(True, dtype=bool)

[](jax.numpy.linalg.slogdet.html "previous page")

previous

jax.numpy.linalg.slogdet

[](jax.numpy.linalg.svd.html "next page")

next

jax.numpy.linalg.svd

Contents

- [`solve()`](#jax.numpy.linalg.solve)

By The JAX authors

© Copyright 2024, The JAX Authors.\
