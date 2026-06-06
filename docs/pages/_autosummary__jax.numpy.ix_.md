- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.ix\_

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.ix_.rst "Download source file")
-  .pdf

# jax.numpy.ix\_

## Contents

- [`ix_()`](#jax.numpy.ix_)

# jax.numpy.ix\_[\#](#jax-numpy-ix "Link to this heading")

jax.numpy.ix\_(*\*args*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L6139-L6191)[\#](#jax.numpy.ix_ "Link to this definition")  
Return a multi-dimensional grid (open mesh) from N one-dimensional sequences.

JAX implementation of [`numpy.ix_()`](https://numpy.org/doc/stable/reference/generated/numpy.ix_.html#numpy.ix_ "(in NumPy v2.4)").

Parameters:  
**\*args** (*ArrayLike*) – N one-dimensional arrays

Returns:  
Tuple of Jax arrays forming an open mesh, each with N dimensions.

Return type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), …\]

See also

- [`jax.numpy.ogrid`](jax.numpy.ogrid.html#jax.numpy.ogrid "jax.numpy.ogrid")

- [`jax.numpy.mgrid`](jax.numpy.mgrid.html#jax.numpy.mgrid "jax.numpy.mgrid")

- [`jax.numpy.meshgrid()`](jax.numpy.meshgrid.html#jax.numpy.meshgrid "jax.numpy.meshgrid")

Examples

    >>> rows = jnp.array([0, 2])
    >>> cols = jnp.array([1, 3])
    >>> open_mesh = jnp.ix_(rows, cols)
    >>> open_mesh
    (Array([[0],
          [2]], dtype=int32), Array([[1, 3]], dtype=int32))
    >>> [grid.shape for grid in open_mesh]
    [(2, 1), (1, 2)]
    >>> x = jnp.array([[10, 20, 30, 40],
    ...                [50, 60, 70, 80],
    ...                [90, 100, 110, 120],
    ...                [130, 140, 150, 160]])
    >>> x[open_mesh]
    Array([[ 20,  40],
           [100, 120]], dtype=int32)

[](jax.numpy.iterable.html "previous page")

previous

jax.numpy.iterable

[](jax.numpy.kaiser.html "next page")

next

jax.numpy.kaiser

Contents

- [`ix_()`](#jax.numpy.ix_)

By The JAX authors

© Copyright 2024, The JAX Authors.\
