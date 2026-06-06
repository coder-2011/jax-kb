- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.linalg.svd

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.linalg.svd.rst "Download source file")
-  .pdf

# jax.scipy.linalg.svd

## Contents

- [`svd()`](#jax.scipy.linalg.svd)

# jax.scipy.linalg.svd[\#](#jax-scipy-linalg-svd "Link to this heading")

jax.scipy.linalg.svd(*a: ArrayLike*, *full_matrices: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = True*, *compute_uv: [Literal](jax.extend.core.Literal.html#jax.extend.core.Literal "jax.extend.core.Literal")\[True\] = True*, *overwrite_a: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = False*, *check_finite: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = True*, *lapack_driver: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") = 'gesdd'*) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\][\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/linalg.py#L273-L344)[\#](#jax.scipy.linalg.svd "Link to this definition")\
jax.scipy.linalg.svd(*a: ArrayLike*, *full_matrices: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")*, *compute_uv: [Literal](jax.extend.core.Literal.html#jax.extend.core.Literal "jax.extend.core.Literal")\[False\]*, *overwrite_a: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = False*, *check_finite: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = True*, *lapack_driver: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") = 'gesdd'*) → [Array](jax.Array.html#jax.Array "jax.Array")\
jax.scipy.linalg.svd(*a: ArrayLike*, *full_matrices: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = True*, *\**, *compute_uv: [Literal](jax.extend.core.Literal.html#jax.extend.core.Literal "jax.extend.core.Literal")\[False\]*, *overwrite_a: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = False*, *check_finite: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = True*, *lapack_driver: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") = 'gesdd'*) → [Array](jax.Array.html#jax.Array "jax.Array")\
jax.scipy.linalg.svd(*a: ArrayLike*, *full_matrices: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = True*, *compute_uv: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = True*, *overwrite_a: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = False*, *check_finite: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = True*, *lapack_driver: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") = 'gesdd'*) → [Array](jax.Array.html#jax.Array "jax.Array") \| [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\]  
Compute the singular value decomposition.

JAX implementation of [`scipy.linalg.svd()`](https://scipy.github.io/devdocs/reference/generated/scipy.linalg.svd.html#scipy.linalg.svd "(in SciPy v1.19.0.dev)").

The SVD of a matrix A is given by

\\A = U\Sigma V^H\\

- \\U\\ contains the left singular vectors and satisfies \\U^HU=I\\

- \\V\\ contains the right singular vectors and satisfies \\V^HV=I\\

- \\\Sigma\\ is a diagonal matrix of singular values.

Parameters:  
- **a** – input array, of shape `(...,`` ``N,`` ``M)`

- **full_matrices** – if True (default) compute the full matrices; i.e. `u` and `vh` have shape `(...,`` ``N,`` ``N)` and `(...,`` ``M,`` ``M)`. If False, then the shapes are `(...,`` ``N,`` ``K)` and `(...,`` ``K,`` ``M)` with `K`` ``=`` ``min(N,`` ``M)`.

- **compute_uv** – if True (default), return the full SVD `(u,`` ``s,`` ``vh)`. If False then return only the singular values `s`.

- **overwrite_a** – unused by JAX

- **check_finite** – unused by JAX

- **lapack_driver** – unused by JAX. If you want to select a non-default SVD driver, please check [`jax.lax.linalg.svd()`](jax.lax.linalg.svd.html#jax.lax.linalg.svd "jax.lax.linalg.svd") which provides such functionality.

Returns:  
A tuple of arrays `(u,`` ``s,`` ``vh)` if `compute_uv` is True, otherwise the array `s`.

- `u`: left singular vectors of shape `(...,`` ``N,`` ``N)` if `full_matrices` is True or `(...,`` ``N,`` ``K)` otherwise.

- `s`: singular values of shape `(...,`` ``K)`

- `vh`: conjugate-transposed right singular vectors of shape `(...,`` ``M,`` ``M)` if `full_matrices` is True or `(...,`` ``K,`` ``M)` otherwise.

where `K`` ``=`` ``min(N,`` ``M)`.

See also

- [`jax.numpy.linalg.svd()`](jax.numpy.linalg.svd.html#jax.numpy.linalg.svd "jax.numpy.linalg.svd"): NumPy-style SVD API

- [`jax.lax.linalg.svd()`](jax.lax.linalg.svd.html#jax.lax.linalg.svd "jax.lax.linalg.svd"): XLA-style SVD API

Examples

Consider the SVD of a small real-valued array:

    >>> x = jnp.array([[1., 2., 3.],
    ...                [6., 5., 4.]])
    >>> u, s, vt = jax.scipy.linalg.svd(x, full_matrices=False)
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

[](jax.scipy.linalg.sqrtm.html "previous page")

previous

jax.scipy.linalg.sqrtm

[](jax.scipy.linalg.toeplitz.html "next page")

next

jax.scipy.linalg.toeplitz

Contents

- [`svd()`](#jax.scipy.linalg.svd)

By The JAX authors

© Copyright 2024, The JAX Authors.\
