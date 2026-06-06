- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.linalg.lstsq

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.linalg.lstsq.rst "Download source file")
-  .pdf

# jax.numpy.linalg.lstsq

## Contents

- [`lstsq()`](#jax.numpy.linalg.lstsq)

# jax.numpy.linalg.lstsq[\#](#jax-numpy-linalg-lstsq "Link to this heading")

jax.numpy.linalg.lstsq(*a*, *b*, *rcond=None*, *\**, *numpy_resid=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/linalg.py#L1490-L1529)[\#](#jax.numpy.linalg.lstsq "Link to this definition")  
Return the least-squares solution to a linear equation.

JAX implementation of [`numpy.linalg.lstsq()`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html#numpy.linalg.lstsq "(in NumPy v2.4)").

Parameters:  
- **a** (*ArrayLike*) – array of shape `(M,`` ``N)` representing the coefficient matrix.

- **b** (*ArrayLike*) – array of shape `(M,)` or `(M,`` ``K)` representing the right-hand side.

- **rcond** ([*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* *None*) – Cut-off ratio for small singular values. Singular values smaller than `rcond`` ``*`` ``largest_singular_value` are treated as zero. If None (default), the optimal value will be used to reduce floating point errors.

- **numpy_resid** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – If True, compute and return residuals in the same way as NumPy’s linalg.lstsq. This is necessary if you want to precisely replicate NumPy’s behavior. If False (default), a more efficient method is used to compute residuals.

Returns:  
Tuple of arrays `(x,`` ``resid,`` ``rank,`` ``s)` where

- `x` is a shape `(N,)` or `(N,`` ``K)` array containing the least-squares solution.

- `resid` is the sum of squared residual of shape `()` or `(K,)`.

- `rank` is the rank of the matrix `a`.

- `s` is the singular values of the matrix `a`.

Return type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\]

Examples

    >>> a = jnp.array([[1, 2],
    ...                [3, 4]])
    >>> b = jnp.array([5, 6])
    >>> x, _, _, _ = jnp.linalg.lstsq(a, b)
    >>> with jnp.printoptions(precision=3):
    ...   print(x)
    [-4.   4.5]

[](jax.numpy.linalg.inv.html "previous page")

previous

jax.numpy.linalg.inv

[](jax.numpy.linalg.matmul.html "next page")

next

jax.numpy.linalg.matmul

Contents

- [`lstsq()`](#jax.numpy.linalg.lstsq)

By The JAX authors

© Copyright 2024, The JAX Authors.\
