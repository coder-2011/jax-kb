- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.meshgrid

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.meshgrid.rst "Download source file")
-  .pdf

# jax.numpy.meshgrid

## Contents

- [`meshgrid()`](#jax.numpy.meshgrid)

# jax.numpy.meshgrid[\#](#jax-numpy-meshgrid "Link to this heading")

jax.numpy.meshgrid(*\*xi*, *copy=True*, *sparse=False*, *indexing='xy'*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L6008-L6087)[\#](#jax.numpy.meshgrid "Link to this definition")  
Construct N-dimensional grid arrays from N 1-dimensional vectors.

JAX implementation of [`numpy.meshgrid()`](https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html#numpy.meshgrid "(in NumPy v2.4)").

Parameters:  
- **xi** (*ArrayLike*) – N arrays to convert to a grid.

- **copy** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – whether to copy the input arrays. JAX supports only `copy=True`, though under JIT compilation the compiler may opt to avoid copies.

- **sparse** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – if False (default), then each returned arrays will be of shape `[len(x1),`` ``len(x2),`` ``...,`` ``len(xN)]`. If False, then returned arrays will be of shape `[1,`` ``1,`` ``...,`` ``len(xi),`` ``...,`` ``1,`` ``1]`.

- **indexing** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – options are `'xy'` for cartesian indexing (default) or `'ij'` for matrix indexing.

Returns:  
A length-N list of grid arrays.

Return type:  
[list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array")\]

See also

- [`jax.numpy.indices()`](jax.numpy.indices.html#jax.numpy.indices "jax.numpy.indices"): generate a grid of indices.

- [`jax.numpy.mgrid`](jax.numpy.mgrid.html#jax.numpy.mgrid "jax.numpy.mgrid"): create a meshgrid using indexing syntax.

- [`jax.numpy.ogrid`](jax.numpy.ogrid.html#jax.numpy.ogrid "jax.numpy.ogrid"): create an open meshgrid using indexing syntax.

Examples

For the following examples, we’ll use these 1D arrays as inputs:

    >>> x = jnp.array([1, 2])
    >>> y = jnp.array([10, 20, 30])

2D cartesian mesh grid:

    >>> x_grid, y_grid = jnp.meshgrid(x, y)
    >>> print(x_grid)
    [[1 2]
     [1 2]
     [1 2]]
    >>> print(y_grid)
    [[10 10]
     [20 20]
     [30 30]]

2D sparse cartesian mesh grid:

    >>> x_grid, y_grid = jnp.meshgrid(x, y, sparse=True)
    >>> print(x_grid)
    [[1 2]]
    >>> print(y_grid)
    [[10]
     [20]
     [30]]

2D matrix-index mesh grid:

    >>> x_grid, y_grid = jnp.meshgrid(x, y, indexing='ij')
    >>> print(x_grid)
    [[1 1 1]
     [2 2 2]]
    >>> print(y_grid)
    [[10 20 30]
     [10 20 30]]

[](jax.numpy.median.html "previous page")

previous

jax.numpy.median

[](jax.numpy.mgrid.html "next page")

next

jax.numpy.mgrid

Contents

- [`meshgrid()`](#jax.numpy.meshgrid)

By The JAX authors

© Copyright 2024, The JAX Authors.\
