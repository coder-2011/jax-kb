- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.vander

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.vander.rst "Download source file")
-  .pdf

# jax.numpy.vander

## Contents

- [`vander()`](#jax.numpy.vander)

# jax.numpy.vander[\#](#jax-numpy-vander "Link to this heading")

jax.numpy.vander(*x*, *N=None*, *increasing=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L8088-L8147)[\#](#jax.numpy.vander "Link to this definition")  
Generate a Vandermonde matrix.

JAX implementation of [`numpy.vander()`](https://numpy.org/doc/stable/reference/generated/numpy.vander.html#numpy.vander "(in NumPy v2.4)").

Parameters:  
- **x** (*ArrayLike*) – input array. Must have `x.ndim`` ``==`` ``1`.

- **N** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – int, optional, default=None. Specifies the number of the columns the output matrix. If not specified, `N`` ``=`` ``len(x)`.

- **increasing** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – bool, optional, default=False. Specifies the order of the powers of the columns. If `True`, the powers increase from left to right, \\\[x^0, x^1, ..., x^{(N-1)}\]\\. By default, the powers decrease from left to right \\\[x^{(N-1)}, ..., x^1, x^0\]\\.

Returns:  
An array of shape `[len(x),`` ``N]` containing the generated Vandermonde matrix.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Examples

    >>> x = jnp.array([1, 2, 3, 4])
    >>> jnp.vander(x)
    Array([[ 1,  1,  1,  1],
           [ 8,  4,  2,  1],
           [27,  9,  3,  1],
           [64, 16,  4,  1]], dtype=int32)

If `N`` ``=`` ``2`, generates a Vandermonde matrix with `2` columns.

    >>> jnp.vander(x, N=2)
    Array([[1, 1],
           [2, 1],
           [3, 1],
           [4, 1]], dtype=int32)

Generates the Vandermonde matrix in increasing order of powers, when `increasing=True`.

    >>> jnp.vander(x, increasing=True)
    Array([[ 1,  1,  1,  1],
           [ 1,  2,  4,  8],
           [ 1,  3,  9, 27],
           [ 1,  4, 16, 64]], dtype=int32)

[](jax.numpy.unwrap.html "previous page")

previous

jax.numpy.unwrap

[](jax.numpy.var.html "next page")

next

jax.numpy.var

Contents

- [`vander()`](#jax.numpy.vander)

By The JAX authors

© Copyright 2024, The JAX Authors.\
