- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.linalg.svd

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.linalg.svd.rst "Download source file")
-  .pdf

# jax.numpy.linalg.svd

## Contents

- [`svd()`](#jax.numpy.linalg.svd)

# jax.numpy.linalg.svd[\#](#jax-numpy-linalg-svd "Link to this heading")

jax.numpy.linalg.svd(*a*, *full_matrices=True*, *compute_uv=True*, *hermitian=False*, *subset_by_index=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/linalg.py#L209-L330)[\#](#jax.numpy.linalg.svd "Link to this definition")  
Compute the singular value decomposition.

JAX implementation of [`numpy.linalg.svd()`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html#numpy.linalg.svd "(in NumPy v2.4)"), implemented in terms of [`jax.lax.linalg.svd()`](jax.lax.linalg.svd.html#jax.lax.linalg.svd "jax.lax.linalg.svd").

The SVD of a matrix A is given by

\\A = U\Sigma V^H\\

- \\U\\ contains the left singular vectors and satisfies \\U^HU=I\\

- \\V\\ contains the right singular vectors and satisfies \\V^HV=I\\

- \\\Sigma\\ is a diagonal matrix of singular values.

Parameters:  
- **a** (*ArrayLike*) – input array, of shape `(...,`` ``N,`` ``M)`

- **full_matrices** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – if True (default) compute the full matrices; i.e. `u` and `vh` have shape `(...,`` ``N,`` ``N)` and `(...,`` ``M,`` ``M)`. If False, then the shapes are `(...,`` ``N,`` ``K)` and `(...,`` ``K,`` ``M)` with `K`` ``=`` ``min(N,`` ``M)`.

- **compute_uv** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – if True (default), return the full SVD `(u,`` ``s,`` ``vh)`. If False then return only the singular values `s`.

- **hermitian** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – if True, assume the matrix is hermitian, which allows for a more efficient implementation (default=False)

- **subset_by_index** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]* *\|* *None*) – (TPU-only) Optional 2-tuple \[start, end\] indicating the range of indices of singular values to compute. For example, if `[n-2,`` ``n]` then `svd` computes the two largest singular values and their singular vectors. Only compatible with `full_matrices=False`.

Returns:  
A tuple of arrays `(u,`` ``s,`` ``vh)` if `compute_uv` is True, otherwise the array `s`.

- `u`: left singular vectors of shape `(...,`` ``N,`` ``N)` if `full_matrices` is True or `(...,`` ``N,`` ``K)` otherwise.

- `s`: singular values of shape `(...,`` ``K)`

- `vh`: conjugate-transposed right singular vectors of shape `(...,`` ``M,`` ``M)` if `full_matrices` is True or `(...,`` ``K,`` ``M)` otherwise.

where `K`` ``=`` ``min(N,`` ``M)`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array") \| SVDResult

See also

- [`jax.scipy.linalg.svd()`](jax.scipy.linalg.svd.html#jax.scipy.linalg.svd "jax.scipy.linalg.svd"): SciPy-style SVD API

- [`jax.lax.linalg.svd()`](jax.lax.linalg.svd.html#jax.lax.linalg.svd "jax.lax.linalg.svd"): XLA-style SVD API

Examples

Consider the SVD of a small real-valued array:

    >>> x = jnp.array([[1., 2., 3.],
    ...                [6., 5., 4.]])
    >>> u, s, vt = jnp.linalg.svd(x, full_matrices=False)
    >>> s  
    Array([9.361919 , 1.8315067], dtype=float32)

The singular vectors are in the columns of `u` and `v`` ``=`` ``vt.T`. These vectors are orthonormal, which can be demonstrated by comparing the matrix product with the identity matrix:

    >>> jnp.allclose(u.T @ u, jnp.eye(2), atol=1E-5)
    Array(True, dtype=bool)
    >>> v = vt.T
    >>> jnp.allclose(v.T @ v, jnp.eye(2), atol=1E-5)
    Array(True, dtype=bool)

Given the SVD, `x` can be reconstructed via matrix multiplication:

    >>> x_reconstructed = u @ jnp.diag(s) @ vt
    >>> jnp.allclose(x_reconstructed, x)
    Array(True, dtype=bool)

[](jax.numpy.linalg.solve.html "previous page")

previous

jax.numpy.linalg.solve

[](jax.numpy.linalg.svdvals.html "next page")

next

jax.numpy.linalg.svdvals

Contents

- [`svd()`](#jax.numpy.linalg.svd)

By The JAX authors

© Copyright 2024, The JAX Authors.\
