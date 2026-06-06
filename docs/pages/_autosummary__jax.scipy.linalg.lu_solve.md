- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.linalg.lu_solve

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.linalg.lu_solve.rst "Download source file")
-  .pdf

# jax.scipy.linalg.lu_solve

## Contents

- [`lu_solve()`](#jax.scipy.linalg.lu_solve)

# jax.scipy.linalg.lu_solve[\#](#jax-scipy-linalg-lu-solve "Link to this heading")

jax.scipy.linalg.lu_solve(*lu_and_piv*, *b*, *trans=0*, *overwrite_b=False*, *check_finite=True*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/linalg.py#L688-L762)[\#](#jax.scipy.linalg.lu_solve "Link to this definition")  
Solve a linear system using an LU factorization

JAX implementation of [`scipy.linalg.lu_solve()`](https://scipy.github.io/devdocs/reference/generated/scipy.linalg.lu_solve.html#scipy.linalg.lu_solve "(in SciPy v1.19.0.dev)"). Uses the output of [`jax.scipy.linalg.lu_factor()`](jax.scipy.linalg.lu_factor.html#jax.scipy.linalg.lu_factor "jax.scipy.linalg.lu_factor").

Parameters:  
- **lu_and_piv** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*Array*](jax.Array.html#jax.Array "jax.Array")*,* *ArrayLike\]*) – `(lu,`` ``piv)`, output of [`lu_factor()`](jax.scipy.linalg.lu_factor.html#jax.scipy.linalg.lu_factor "jax.scipy.linalg.lu_factor"). `lu` is an array of shape `(...,`` ``M,`` ``N)`, containing `L` in its lower triangle and `U` in its upper. `piv` is an array of shape `(...,`` ``K)`, with `K`` ``=`` ``min(M,`` ``N)`, which encodes the pivots.

- **b** (*ArrayLike*) – right-hand-side of linear system. Array of shape `(M,)` (for a 1-dimensional right-hand-side) or `(...,`` ``M,`` ``K)` (for a batched 2-dimensional right-hand-side).

- **trans** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) –

  type of system to solve. Options are:

  - `0`: \\A x = b\\

  - `1`: \\A^Tx = b\\

  - `2`: \\A^Hx = b\\

- **overwrite_b** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – unused by JAX

- **check_finite** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – unused by JAX

Returns:  
Array representing the solution of the linear system. The result has shape `(...,`` ``N)` if `b` is of shape `(M,)`, and has shape `(...,`` ``N,`` ``K)` otherwise.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.scipy.linalg.lu()`](jax.scipy.linalg.lu.html#jax.scipy.linalg.lu "jax.scipy.linalg.lu")

- [`jax.scipy.linalg.lu_factor()`](jax.scipy.linalg.lu_factor.html#jax.scipy.linalg.lu_factor "jax.scipy.linalg.lu_factor")

Examples

Solving a small linear system via LU factorization:

    >>> a = jnp.array([[2., 1.],
    ...                [1., 2.]])

Compute the lu factorization via [`lu_factor()`](jax.scipy.linalg.lu_factor.html#jax.scipy.linalg.lu_factor "jax.scipy.linalg.lu_factor"), and use it to solve a linear equation via [`lu_solve()`](#jax.scipy.linalg.lu_solve "jax.scipy.linalg.lu_solve").

    >>> b = jnp.array([3., 4.])
    >>> lufac = jax.scipy.linalg.lu_factor(a)
    >>> y = jax.scipy.linalg.lu_solve(lufac, b)
    >>> y
    Array([0.6666666, 1.6666667], dtype=float32)

Check that the result is consistent:

    >>> jnp.allclose(a @ y, b)
    Array(True, dtype=bool)

[](jax.scipy.linalg.lu_factor.html "previous page")

previous

jax.scipy.linalg.lu_factor

[](jax.scipy.linalg.pascal.html "next page")

next

jax.scipy.linalg.pascal

Contents

- [`lu_solve()`](#jax.scipy.linalg.lu_solve)

By The JAX authors

© Copyright 2024, The JAX Authors.\
