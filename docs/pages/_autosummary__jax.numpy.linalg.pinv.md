- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.linalg.pinv

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.linalg.pinv.rst "Download source file")
-  .pdf

# jax.numpy.linalg.pinv

## Contents

- [`pinv()`](#jax.numpy.linalg.pinv)

# jax.numpy.linalg.pinv[\#](#jax-numpy-linalg-pinv "Link to this heading")

jax.numpy.linalg.pinv(*a*, *rtol=None*, *hermitian=False*, *\**, *rcond=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/linalg.py#L984-L1035)[\#](#jax.numpy.linalg.pinv "Link to this definition")  
Compute the (Moore-Penrose) pseudo-inverse of a matrix.

JAX implementation of [`numpy.linalg.pinv()`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.pinv.html#numpy.linalg.pinv "(in NumPy v2.4)").

Parameters:  
- **a** (*ArrayLike*) – array of shape `(...,`` ``M,`` ``N)` containing matrices to pseudo-invert.

- **rtol** (*ArrayLike* *\|* *None*) – float or array_like of shape `a.shape[:-2]`. Specifies the cutoff for small singular values.of shape `(...,)`. Cutoff for small singular values; singular values smaller `rtol`` ``*`` ``largest_singular_value` are treated as zero. The default is determined based on the floating point precision of the dtype.

- **hermitian** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – if True, then the input is assumed to be Hermitian, and a more efficient algorithm is used (default: False)

- **rcond** (*ArrayLike* *\|* *None*) – alias of the rtol argument, present for backward compatibility. Only one of rtol and rcond may be specified.

Returns:  
An array of shape `(...,`` ``N,`` ``M)` containing the pseudo-inverse of `a`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.linalg.inv()`](jax.numpy.linalg.inv.html#jax.numpy.linalg.inv "jax.numpy.linalg.inv"): multiplicative inverse of a square matrix.

Notes

[`jax.numpy.linalg.pinv()`](#jax.numpy.linalg.pinv "jax.numpy.linalg.pinv") differs from [`numpy.linalg.pinv()`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.pinv.html#numpy.linalg.pinv "(in NumPy v2.4)") in the default value of rcond\`: in NumPy, the default is 1e-15. In JAX, the default is `10.`` ``*`` ``max(num_rows,`` ``num_cols)`` ``*`` ``jnp.finfo(dtype).eps`.

Examples

    >>> a = jnp.array([[1, 2],
    ...                [3, 4],
    ...                [5, 6]])
    >>> a_pinv = jnp.linalg.pinv(a)
    >>> a_pinv  
    Array([[-1.333332  , -0.33333257,  0.6666657 ],
           [ 1.0833322 ,  0.33333272, -0.41666582]], dtype=float32)

The pseudo-inverse operates as a multiplicative inverse so long as the output is not rank-deficient:

    >>> jnp.allclose(a_pinv @ a, jnp.eye(2), atol=1E-4)
    Array(True, dtype=bool)

[](jax.numpy.linalg.outer.html "previous page")

previous

jax.numpy.linalg.outer

[](jax.numpy.linalg.qr.html "next page")

next

jax.numpy.linalg.qr

Contents

- [`pinv()`](#jax.numpy.linalg.pinv)

By The JAX authors

© Copyright 2024, The JAX Authors.\
