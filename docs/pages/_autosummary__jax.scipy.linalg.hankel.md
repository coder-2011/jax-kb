- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.linalg.hankel

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.linalg.hankel.rst "Download source file")
-  .pdf

# jax.scipy.linalg.hankel

## Contents

- [`hankel()`](#jax.scipy.linalg.hankel)

# jax.scipy.linalg.hankel[\#](#jax-scipy-linalg-hankel "Link to this heading")

jax.scipy.linalg.hankel(*c*, *r=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/linalg.py#L2507-L2559)[\#](#jax.scipy.linalg.hankel "Link to this definition")  
Construct a Hankel matrix.

JAX implementation of [`scipy.linalg.hankel()`](https://scipy.github.io/devdocs/reference/generated/scipy.linalg.hankel.html#scipy.linalg.hankel "(in SciPy v1.19.0.dev)").

A Hankel matrix has constant anti-diagonals: `H[i,`` ``j]`` ``=`` ``v[i`` ``+`` ``j]`, where `v`` ``=`` ``concatenate([c,`` ``r[1:]])`. Notice this implies that `r[0]` is ignored.

Parameters:  
- **c** (*ArrayLike*) – array of shape `(...,`` ``N)` specifying the first column.

- **r** (*ArrayLike* *\|* *None*) – (optional) array of shape `(...,`` ``M)` specifying the last row. Leading dimensions must be broadcast-compatible with those of `c`. If not specified, `r` defaults to `zeros_like(c)`.

Returns:  
A Hankel matrix of shape `(...,`` ``N,`` ``M)`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Examples

    >>> c = jnp.array([1, 2, 3])
    >>> jax.scipy.linalg.hankel(c)
    Array([[1, 2, 3],
           [2, 3, 0],
           [3, 0, 0]], dtype=int32)

    >>> r = jnp.array([999, 4, 5, 6]) # Note r[0] is ignored
    >>> jax.scipy.linalg.hankel(c, r)
    Array([[1, 2, 3, 4],
           [2, 3, 4, 5],
           [3, 4, 5, 6]], dtype=int32)

For N-dimensional `c` and/or `r`, the result is a batch of Hankel matrices.

[](jax.scipy.linalg.hessenberg.html "previous page")

previous

jax.scipy.linalg.hessenberg

[](jax.scipy.linalg.hilbert.html "next page")

next

jax.scipy.linalg.hilbert

Contents

- [`hankel()`](#jax.scipy.linalg.hankel)

By The JAX authors

© Copyright 2024, The JAX Authors.\
