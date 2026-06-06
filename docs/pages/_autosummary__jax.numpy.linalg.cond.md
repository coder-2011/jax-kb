- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.linalg.cond

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.linalg.cond.rst "Download source file")
-  .pdf

# jax.numpy.linalg.cond

## Contents

- [`cond()`](#jax.numpy.linalg.cond)

# jax.numpy.linalg.cond[\#](#jax-numpy-linalg-cond "Link to this heading")

jax.numpy.linalg.cond(*x*, *p=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/linalg.py#L2204-L2261)[\#](#jax.numpy.linalg.cond "Link to this definition")  
Compute the condition number of a matrix.

JAX implementation of [`numpy.linalg.cond()`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.cond.html#numpy.linalg.cond "(in NumPy v2.4)").

The condition number is defined as `norm(x,`` ``p)`` ``*`` ``norm(inv(x),`` ``p)`. For `p`` ``=`` ``2` (the default), the condition number is the ratio of the largest to the smallest singular value.

Parameters:  
- **x** (*ArrayLike*) – array of shape `(...,`` ``M,`` ``N)` for which to compute the condition number.

- **p** – the order of the norm to use. One of `{None,`` ``1,`` ``-1,`` ``2,`` ``-2,`` ``inf,`` ``-inf,`` ``'fro'}`; see [`jax.numpy.linalg.norm()`](jax.numpy.linalg.norm.html#jax.numpy.linalg.norm "jax.numpy.linalg.norm") for the meaning of these. The default is `p`` ``=`` ``None`, which is equivalent to `p`` ``=`` ``2`. If not in `{None,`` ``2,`` ``-2}` then `x` must be square, i.e. `M`` ``=`` ``N`.

Returns:  
array of shape `x.shape[:-2]` containing the condition number.

See also

[`jax.numpy.linalg.norm()`](jax.numpy.linalg.norm.html#jax.numpy.linalg.norm "jax.numpy.linalg.norm")

Examples

Well-conditioned matrix:

    >>> x = jnp.array([[1, 2],
    ...                [2, 1]])
    >>> jnp.linalg.cond(x)
    Array(3., dtype=float32)

Ill-conditioned matrix:

    >>> x = jnp.array([[1, 2],
    ...                [0, 0]])
    >>> jnp.linalg.cond(x)
    Array(inf, dtype=float32)

[](jax.numpy.linalg.cholesky.html "previous page")

previous

jax.numpy.linalg.cholesky

[](jax.numpy.linalg.cross.html "next page")

next

jax.numpy.linalg.cross

Contents

- [`cond()`](#jax.numpy.linalg.cond)

By The JAX authors

© Copyright 2024, The JAX Authors.\
