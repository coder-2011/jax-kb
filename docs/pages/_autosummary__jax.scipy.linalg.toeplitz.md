- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.linalg.toeplitz

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.linalg.toeplitz.rst "Download source file")
-  .pdf

# jax.scipy.linalg.toeplitz

## Contents

- [`toeplitz()`](#jax.scipy.linalg.toeplitz)

# jax.scipy.linalg.toeplitz[\#](#jax-scipy-linalg-toeplitz "Link to this heading")

jax.scipy.linalg.toeplitz(*c*, *r=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/linalg.py#L2417-L2492)[\#](#jax.scipy.linalg.toeplitz "Link to this definition")  
Construct a Toeplitz matrix.

JAX implementation of [`scipy.linalg.toeplitz()`](https://scipy.github.io/devdocs/reference/generated/scipy.linalg.toeplitz.html#scipy.linalg.toeplitz "(in SciPy v1.19.0.dev)").

A Toeplitz matrix has equal diagonals: \\A\_{ij} = k\_{i - j}\\ for \\0 \le i \< n\\ and \\0 \le j \< n\\. This function specifies the diagonals via the first column `c` and the first row `r`, such that for row i and column j:

\\\begin{split}A\_{ij} = \begin{cases} c\_{i - j} & i \ge j \\ r\_{j - i} & i \< j \end{cases}\end{split}\\

Notice this implies that \\r_0\\ is ignored.

Parameters:  
- **c** (*ArrayLike*) – array of shape `(...,`` ``N)` specifying the first column.

- **r** (*ArrayLike* *\|* *None*) – (optional) array of shape `(...,`` ``M)` specifying the first row. Leading dimensions must be broadcast-compatible with those of `c`. If not specified, `r` defaults to `conj(c)`.

Returns:  
A Toeplitz matrix of shape `(...`` ``N,`` ``M)`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Examples

Specifying `c` only:

    >>> c = jnp.array([1, 2, 3])
    >>> jax.scipy.linalg.toeplitz(c)
    Array([[1, 2, 3],
           [2, 1, 2],
           [3, 2, 1]], dtype=int32)

Specifying `c` and `r`:

    >>> r = jnp.array([-1, -2, -3])
    >>> jax.scipy.linalg.toeplitz(c, r)  # Note r[0] is ignored
    Array([[ 1, -2, -3],
           [ 2,  1, -2],
           [ 3,  2,  1]], dtype=int32)

If specifying only complex-valued `c`, `r` defaults to `c.conj()`, resulting in a Hermitian matrix if `c[0].imag`` ``==`` ``0`:

    >>> c = jnp.array([1, 2+1j, 1+2j])
    >>> M = jax.scipy.linalg.toeplitz(c)
    >>> M
    Array([[1.+0.j, 2.-1.j, 1.-2.j],
           [2.+1.j, 1.+0.j, 2.-1.j],
           [1.+2.j, 2.+1.j, 1.+0.j]], dtype=complex64)
    >>> print("M is Hermitian:", jnp.all(M == M.conj().T))
    M is Hermitian: True

For N-dimensional `c` and/or `r`, the result is a batch of Toeplitz matrices:

    >>> c = jnp.array([[1, 2, 3], [4, 5, 6]])
    >>> jax.scipy.linalg.toeplitz(c)
    Array([[[1, 2, 3],
            [2, 1, 2],
            [3, 2, 1]],

           [[4, 5, 6],
            [5, 4, 5],
            [6, 5, 4]]], dtype=int32)

[](jax.scipy.linalg.svd.html "previous page")

previous

jax.scipy.linalg.svd

[](jax.scipy.ndimage.map_coordinates.html "next page")

next

jax.scipy.ndimage.map_coordinates

Contents

- [`toeplitz()`](#jax.scipy.linalg.toeplitz)

By The JAX authors

© Copyright 2024, The JAX Authors.\
