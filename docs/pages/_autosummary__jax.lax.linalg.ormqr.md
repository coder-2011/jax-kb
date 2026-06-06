- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.linalg.ormqr

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.linalg.ormqr.rst "Download source file")
-  .pdf

# jax.lax.linalg.ormqr

## Contents

- [`ormqr()`](#jax.lax.linalg.ormqr)

# jax.lax.linalg.ormqr[\#](#jax-lax-linalg-ormqr "Link to this heading")

jax.lax.linalg.ormqr(*a*, *taus*, *c*, *\**, *left=True*, *transpose=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/linalg.py#L1471-L1518)[\#](#jax.lax.linalg.ormqr "Link to this definition")  
Multiplies a matrix by Q from a QR factorization without materializing Q.

Computes `Q`` ``@`` ``C` (`left=True`, `transpose=False`), `Q^T`` ``@`` ``C` (`left=True`, `transpose=True`), `C`` ``@`` ``Q` (`left=False`, `transpose=False`), or `C`` ``@`` ``Q^T` (`left=False`, `transpose=True`).

For complex types, `transpose=True` computes the conjugate transpose (`Q^H`).

Parameters:  
- **a** (*ArrayLike*) – The Householder reflectors with shape `[...,`` ``m,`` ``n]`, as returned by the internal `geqrf`/`geqp3` primitives. Alternatively, one can use [`jax.numpy.linalg.qr()`](jax.numpy.linalg.qr.html#jax.numpy.linalg.qr "jax.numpy.linalg.qr") with `mode="raw"`, but in this case the returned `a` must be transposed with `.mT` (see example below).

- **taus** (*ArrayLike*) – The Householder scalar factors with shape `[...,`` ``k]`, as returned by `geqrf`/`geqp3` or the second element of the tuple from [`jax.numpy.linalg.qr()`](jax.numpy.linalg.qr.html#jax.numpy.linalg.qr "jax.numpy.linalg.qr") with `mode="raw"`.

- **c** (*ArrayLike*) – The matrix to multiply by Q, with shape `[...,`` ``c_rows,`` ``c_cols]`.

- **left** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – If `True`, compute `Q`` ``@`` ``C`. If `False`, compute `C`` ``@`` ``Q`.

- **transpose** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – If `True`, use `Q^T` (or `Q^H` for complex types).

Returns:  
The result of multiplying `c` by Q (or `Q^T`/`Q^H`), with the same shape as `c`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Examples

Multiply a vector by Q without forming Q explicitly:

    >>> import jax.numpy as jnp
    >>> from jax.lax.linalg import ormqr
    >>> a = jnp.array([[1., 2.], [3., 4.], [5., 6.]])
    >>> h, taus = jnp.linalg.qr(a, mode="raw")
    >>> c = jnp.eye(3)
    >>> Q_times_c = ormqr(h.mT, taus, c)
    >>> Q_direct, _ = jnp.linalg.qr(a, mode="complete")
    >>> jnp.allclose(Q_times_c, Q_direct, atol=1e-5)
    Array(True, dtype=bool)

See also

- [`jax.scipy.linalg.qr_multiply()`](jax.scipy.linalg.qr_multiply.html#jax.scipy.linalg.qr_multiply "jax.scipy.linalg.qr_multiply"): Higher-level API for computing Q @ C or C @ Q from a matrix `a` directly.

[](jax.lax.linalg.lu_pivots_to_permutation.html "previous page")

previous

jax.lax.linalg.lu_pivots_to_permutation

[](jax.lax.linalg.qdwh.html "next page")

next

jax.lax.linalg.qdwh

Contents

- [`ormqr()`](#jax.lax.linalg.ormqr)

By The JAX authors

© Copyright 2024, The JAX Authors.\
