- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.linalg.lu_factor

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.linalg.lu_factor.rst "Download source file")
-  .pdf

# jax.scipy.linalg.lu_factor

## Contents

- [`lu_factor()`](#jax.scipy.linalg.lu_factor)

# jax.scipy.linalg.lu_factor[\#](#jax-scipy-linalg-lu-factor "Link to this heading")

jax.scipy.linalg.lu_factor(*a*, *overwrite_a=False*, *check_finite=True*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/linalg.py#L636-L686)[\#](#jax.scipy.linalg.lu_factor "Link to this definition")  
Factorization for LU-based linear solves

JAX implementation of [`scipy.linalg.lu_factor()`](https://scipy.github.io/devdocs/reference/generated/scipy.linalg.lu_factor.html#scipy.linalg.lu_factor "(in SciPy v1.19.0.dev)").

This function returns a result suitable for use with [`jax.scipy.linalg.lu_solve()`](jax.scipy.linalg.lu_solve.html#jax.scipy.linalg.lu_solve "jax.scipy.linalg.lu_solve"). For direct LU decompositions, prefer [`jax.scipy.linalg.lu()`](jax.scipy.linalg.lu.html#jax.scipy.linalg.lu "jax.scipy.linalg.lu").

Parameters:  
- **a** (*ArrayLike*) – input array of shape `(...,`` ``M,`` ``N)`.

- **overwrite_a** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – unused by JAX

- **check_finite** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – unused by JAX

Returns:  
A tuple `(lu,`` ``piv)`

- `lu` is an array of shape `(...,`` ``M,`` ``N)`, containing `L` in its lower triangle and `U` in its upper.

- `piv` is an array of shape `(...,`` ``K)` with `K`` ``=`` ``min(M,`` ``N)`, which encodes the pivots.

Return type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\]

See also

- [`jax.scipy.linalg.lu()`](jax.scipy.linalg.lu.html#jax.scipy.linalg.lu "jax.scipy.linalg.lu")

- [`jax.scipy.linalg.lu_solve()`](jax.scipy.linalg.lu_solve.html#jax.scipy.linalg.lu_solve "jax.scipy.linalg.lu_solve")

Examples

Solving a small linear system via LU factorization:

    >>> a = jnp.array([[2., 1.],
    ...                [1., 2.]])

Compute the lu factorization via [`lu_factor()`](#jax.scipy.linalg.lu_factor "jax.scipy.linalg.lu_factor"), and use it to solve a linear equation via [`lu_solve()`](jax.scipy.linalg.lu_solve.html#jax.scipy.linalg.lu_solve "jax.scipy.linalg.lu_solve").

    >>> b = jnp.array([3., 4.])
    >>> lufac = jax.scipy.linalg.lu_factor(a)
    >>> y = jax.scipy.linalg.lu_solve(lufac, b)
    >>> y
    Array([0.6666666, 1.6666667], dtype=float32)

Check that the result is consistent:

    >>> jnp.allclose(a @ y, b)
    Array(True, dtype=bool)

[](jax.scipy.linalg.lu.html "previous page")

previous

jax.scipy.linalg.lu

[](jax.scipy.linalg.lu_solve.html "next page")

next

jax.scipy.linalg.lu_solve

Contents

- [`lu_factor()`](#jax.scipy.linalg.lu_factor)

By The JAX authors

© Copyright 2024, The JAX Authors.\
