- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.linalg.svdvals

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.linalg.svdvals.rst "Download source file")
-  .pdf

# jax.numpy.linalg.svdvals

## Contents

- [`svdvals()`](#jax.numpy.linalg.svdvals)

# jax.numpy.linalg.svdvals[\#](#jax-numpy-linalg-svdvals "Link to this heading")

jax.numpy.linalg.svdvals(*x*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/linalg.py#L1955-L1978)[\#](#jax.numpy.linalg.svdvals "Link to this definition")  
Compute the singular values of a matrix.

JAX implementation of [`numpy.linalg.svdvals()`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svdvals.html#numpy.linalg.svdvals "(in NumPy v2.4)").

Parameters:  
**x** (*ArrayLike*) – array of shape `(...,`` ``M,`` ``N)` for which singular values will be computed.

Returns:  
array of singular values of shape `(...,`` ``K)` with `K`` ``=`` ``min(M,`` ``N)`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

[`jax.numpy.linalg.svd()`](jax.numpy.linalg.svd.html#jax.numpy.linalg.svd "jax.numpy.linalg.svd"): compute singular values and singular vectors

Examples

    >>> x = jnp.array([[1, 2, 3],
    ...                [4, 5, 6]])
    >>> jnp.linalg.svdvals(x)
    Array([9.508031 , 0.7728694], dtype=float32)

[](jax.numpy.linalg.svd.html "previous page")

previous

jax.numpy.linalg.svd

[](jax.numpy.linalg.tensordot.html "next page")

next

jax.numpy.linalg.tensordot

Contents

- [`svdvals()`](#jax.numpy.linalg.svdvals)

By The JAX authors

© Copyright 2024, The JAX Authors.\
