- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.linalg.eigh_tridiagonal

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.linalg.eigh_tridiagonal.rst "Download source file")
-  .pdf

# jax.scipy.linalg.eigh_tridiagonal

## Contents

- [`eigh_tridiagonal()`](#jax.scipy.linalg.eigh_tridiagonal)

# jax.scipy.linalg.eigh_tridiagonal[\#](#jax-scipy-linalg-eigh-tridiagonal "Link to this heading")

jax.scipy.linalg.eigh_tridiagonal(*d*, *e*, *\**, *eigvals_only=False*, *select='a'*, *select_range=None*, *tol=None*, *key=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/linalg.py#L1675-L2014)[\#](#jax.scipy.linalg.eigh_tridiagonal "Link to this definition")  
Solve the eigenvalue problem for a symmetric real tridiagonal matrix

JAX implementation of [`scipy.linalg.eigh_tridiagonal()`](https://scipy.github.io/devdocs/reference/generated/scipy.linalg.eigh_tridiagonal.html#scipy.linalg.eigh_tridiagonal "(in SciPy v1.19.0.dev)").

Parameters:  
- **d** (*ArrayLike*) – real-valued array of shape `(N,)` specifying the diagonal elements.

- **e** (*ArrayLike*) – real-valued array of shape `(N`` ``-`` ``1,)` specifying the off-diagonal elements.

- **eigvals_only** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – If True, return only the eigenvalues (default: False). Computation of eigenvectors is not yet implemented, so `eigvals_only` must be set to True.

- **select** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) –

  specify which eigenvalues to calculate. Supported values are:

  - `'a'`: all eigenvalues

  - `'i'`: eigenvalues with indices `select_range[0]`` ``<=`` ``i`` ``<=`` ``select_range[1]`

  JAX does not currently implement `select`` ``=`` ``'v'`.

- **select_range** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)")*,* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)")*\]* *\|* *None*) – range of values used when `select='i'`.

- **tol** ([*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* *None*) – absolute tolerance to use when solving for the eigenvalues.

- **key** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* *None*) – a PRNG key, as returned by `jax.random.key`, used to generate random initialization vectors for inverse iteration. If `None`, defaults to a fixed PRNG key.

Returns:  
An array of eigenvalues with shape `(N,)`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array") \| [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\]

See also

[`jax.scipy.linalg.eigh()`](jax.scipy.linalg.eigh.html#jax.scipy.linalg.eigh "jax.scipy.linalg.eigh"): general Hermitian eigenvalue solver

Examples

    >>> d = jnp.array([1., 2., 3., 4.])
    >>> e = jnp.array([1., 1., 1.])
    >>> eigvals = jax.scipy.linalg.eigh_tridiagonal(d, e, eigvals_only=True)
    >>> eigvals
    Array([0.2547188, 1.8227171, 3.1772828, 4.745281 ], dtype=float32)

For comparison, we can construct the full matrix and compute the same result using [`eigh()`](jax.scipy.linalg.eigh.html#jax.scipy.linalg.eigh "jax.scipy.linalg.eigh"):

    >>> A = jnp.diag(d) + jnp.diag(e, 1) + jnp.diag(e, -1)
    >>> A
    Array([[1., 1., 0., 0.],
           [1., 2., 1., 0.],
           [0., 1., 3., 1.],
           [0., 0., 1., 4.]], dtype=float32)
    >>> eigvals_full = jax.scipy.linalg.eigh(A, eigvals_only=True)
    >>> jnp.allclose(eigvals, eigvals_full)
    Array(True, dtype=bool)

[](jax.scipy.linalg.eigh.html "previous page")

previous

jax.scipy.linalg.eigh

[](jax.scipy.linalg.expm.html "next page")

next

jax.scipy.linalg.expm

Contents

- [`eigh_tridiagonal()`](#jax.scipy.linalg.eigh_tridiagonal)

By The JAX authors

© Copyright 2024, The JAX Authors.\
