- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.linalg.slogdet

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.linalg.slogdet.rst "Download source file")
-  .pdf

# jax.numpy.linalg.slogdet

## Contents

- [`slogdet()`](#jax.numpy.linalg.slogdet)

# jax.numpy.linalg.slogdet[\#](#jax-numpy-linalg-slogdet "Link to this heading")

jax.numpy.linalg.slogdet(*a*, *\**, *method=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/linalg.py#L575-L623)[\#](#jax.numpy.linalg.slogdet "Link to this definition")  
Compute the sign and (natural) logarithm of the determinant of an array.

JAX implementation of [`numpy.linalg.slogdet()`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.slogdet.html#numpy.linalg.slogdet "(in NumPy v2.4)").

Parameters:  
- **a** (*ArrayLike*) – array of shape `(...,`` ``M,`` ``M)` for which to compute the sign and log determinant.

- **method** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) –

  the method to use for determinant computation. Options are

  - `'lu'` (default): use the LU decomposition.

  - `'qr'`: use the QR decomposition.

Returns:  
A tuple of arrays `(sign,`` ``logabsdet)`, each of shape `a.shape[:-2]`

- `sign` is the sign of the determinant.

- `logabsdet` is the natural log of the determinant’s absolute value.

Return type:  
SlogdetResult

See also

[`jax.numpy.linalg.det()`](jax.numpy.linalg.det.html#jax.numpy.linalg.det "jax.numpy.linalg.det"): direct computation of determinant

Examples

    >>> a = jnp.array([[1, 2],
    ...                [3, 4]])
    >>> sign, logabsdet = jnp.linalg.slogdet(a)
    >>> sign  # -1 indicates negative determinant
    Array(-1., dtype=float32)
    >>> jnp.exp(logabsdet)  # Absolute value of determinant
    Array(2., dtype=float32)

[](jax.numpy.linalg.qr.html "previous page")

previous

jax.numpy.linalg.qr

[](jax.numpy.linalg.solve.html "next page")

next

jax.numpy.linalg.solve

Contents

- [`slogdet()`](#jax.numpy.linalg.slogdet)

By The JAX authors

© Copyright 2024, The JAX Authors.\
