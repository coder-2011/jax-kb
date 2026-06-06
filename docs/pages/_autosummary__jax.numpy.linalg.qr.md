- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.linalg.qr

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.linalg.qr.rst "Download source file")
-  .pdf

# jax.numpy.linalg.qr

## Contents

- [`qr()`](#jax.numpy.linalg.qr)

# jax.numpy.linalg.qr[\#](#jax-numpy-linalg-qr "Link to this heading")

jax.numpy.linalg.qr(*a*, *mode='reduced'*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/linalg.py#L1295-L1377)[\#](#jax.numpy.linalg.qr "Link to this definition")  
Compute the QR decomposition of an array

JAX implementation of [`numpy.linalg.qr()`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.qr.html#numpy.linalg.qr "(in NumPy v2.4)").

The QR decomposition of a matrix A is given by

\\A = QR\\

Where Q is a unitary matrix (i.e. \\Q^HQ=I\\) and R is an upper-triangular matrix.

Parameters:  
- **a** (*ArrayLike*) – array of shape (…, M, N)

- **mode** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) –

  Computational mode. Supported values are:

  - `"reduced"` (default): return Q of shape `(...,`` ``M,`` ``K)` and R of shape `(...,`` ``K,`` ``N)`, where `K`` ``=`` ``min(M,`` ``N)`.

  - `"complete"`: return Q of shape `(...,`` ``M,`` ``M)` and R of shape `(...,`` ``M,`` ``N)`.

  - `"raw"`: return lapack-internal representations of shape `(...,`` ``M,`` ``N)` and `(...,`` ``K)`.

  - `"r"`: return R only.

Returns:  
A tuple `(Q,`` ``R)` (if `mode` is not `"r"`) otherwise an array `R`, where:

- `Q` is an orthogonal matrix of shape `(...,`` ``M,`` ``K)` (if `mode` is `"reduced"`) or `(...,`` ``M,`` ``M)` (if `mode` is `"complete"`).

- `R` is an upper-triangular matrix of shape `(...,`` ``M,`` ``N)` (if `mode` is `"r"` or `"complete"`) or `(...,`` ``K,`` ``N)` (if `mode` is `"reduced"`)

with `K`` ``=`` ``min(M,`` ``N)`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array") \| QRResult

See also

- [`jax.scipy.linalg.qr()`](jax.scipy.linalg.qr.html#jax.scipy.linalg.qr "jax.scipy.linalg.qr"): SciPy-style QR decomposition API

- [`jax.lax.linalg.qr()`](jax.lax.linalg.qr.html#jax.lax.linalg.qr "jax.lax.linalg.qr"): XLA-style QR decomposition API

Examples

Compute the QR decomposition of a matrix:

    >>> a = jnp.array([[1., 2., 3., 4.],
    ...                [5., 4., 2., 1.],
    ...                [6., 3., 1., 5.]])
    >>> Q, R = jnp.linalg.qr(a)
    >>> Q  
    Array([[-0.12700021, -0.7581426 , -0.6396022 ],
           [-0.63500065, -0.43322435,  0.63960224],
           [-0.7620008 ,  0.48737738, -0.42640156]], dtype=float32)
    >>> R  
    Array([[-7.8740077, -5.080005 , -2.4130025, -4.953006 ],
           [ 0.       , -1.7870499, -2.6534991, -1.028908 ],
           [ 0.       ,  0.       , -1.0660033, -4.050814 ]], dtype=float32)

Check that `Q` is orthonormal:

    >>> jnp.allclose(Q.T @ Q, jnp.eye(3), atol=1E-5)
    Array(True, dtype=bool)

Reconstruct the input:

    >>> jnp.allclose(Q @ R, a)
    Array(True, dtype=bool)

[](jax.numpy.linalg.pinv.html "previous page")

previous

jax.numpy.linalg.pinv

[](jax.numpy.linalg.slogdet.html "next page")

next

jax.numpy.linalg.slogdet

Contents

- [`qr()`](#jax.numpy.linalg.qr)

By The JAX authors

© Copyright 2024, The JAX Authors.\
