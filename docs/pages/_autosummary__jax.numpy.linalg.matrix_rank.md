- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.linalg.matrix_rank

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.linalg.matrix_rank.rst "Download source file")
-  .pdf

# jax.numpy.linalg.matrix_rank

## Contents

- [`matrix_rank()`](#jax.numpy.linalg.matrix_rank)

# jax.numpy.linalg.matrix_rank[\#](#jax-numpy-linalg-matrix-rank "Link to this heading")

jax.numpy.linalg.matrix_rank(*M*, *rtol=None*, *\**, *hermitian=False*, *tol=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/linalg.py#L414-L470)[\#](#jax.numpy.linalg.matrix_rank "Link to this definition")  
Compute the rank of a matrix.

JAX implementation of [`numpy.linalg.matrix_rank()`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_rank.html#numpy.linalg.matrix_rank "(in NumPy v2.4)").

The rank is calculated via the Singular Value Decomposition (SVD), and determined by the number of singular values greater than the specified tolerance.

Parameters:  
- **M** (*ArrayLike*) – array of shape `(...,`` ``N,`` ``K)` whose rank is to be computed.

- **rtol** (*ArrayLike* *\|* *None*) – optional array of shape `(...)` specifying the tolerance. Singular values smaller than rtol \* largest_singular_value are considered to be zero. If `rtol` is None (the default), a reasonable default is chosen based the floating point precision of the input.

- **hermitian** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – if True, then the input is assumed to be Hermitian, and a more efficient algorithm is used (default: False)

- **tol** (*ArrayLike* *\|* *None*) – alias of the `rtol` argument present for backward compatibility. Only one of rtol or tol may be specified.

Returns:  
array of shape `a.shape[-2]` giving the matrix rank.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Notes

The rank calculation may be inaccurate for matrices with very small singular values or those that are numerically ill-conditioned. Consider adjusting the `rtol` parameter or using a more specialized rank computation method in such cases.

Examples

    >>> a = jnp.array([[1, 2],
    ...                [3, 4]])
    >>> jnp.linalg.matrix_rank(a)
    Array(2, dtype=int32)

    >>> b = jnp.array([[1, 0],  # Rank-deficient matrix
    ...                [0, 0]])
    >>> jnp.linalg.matrix_rank(b)
    Array(1, dtype=int32)

[](jax.numpy.linalg.matrix_power.html "previous page")

previous

jax.numpy.linalg.matrix_power

[](jax.numpy.linalg.matrix_transpose.html "next page")

next

jax.numpy.linalg.matrix_transpose

Contents

- [`matrix_rank()`](#jax.numpy.linalg.matrix_rank)

By The JAX authors

© Copyright 2024, The JAX Authors.\
