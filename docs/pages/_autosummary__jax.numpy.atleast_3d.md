- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.atleast_3d

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.atleast_3d.rst "Download source file")
-  .pdf

# jax.numpy.atleast_3d

## Contents

- [`atleast_3d()`](#jax.numpy.atleast_3d)

# jax.numpy.atleast_3d[\#](#jax-numpy-atleast-3d "Link to this heading")

jax.numpy.atleast_3d(*\*arys*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L5190-L5255)[\#](#jax.numpy.atleast_3d "Link to this definition")  
Convert inputs to arrays with at least 3 dimensions.

JAX implementation of [`numpy.atleast_3d()`](https://numpy.org/doc/stable/reference/generated/numpy.atleast_3d.html#numpy.atleast_3d "(in NumPy v2.4)").

Parameters:  
- **arguments.** (*zero* *or* *more arraylike*)

- **arys** (*ArrayLike*)

Returns:  
an array or list of arrays corresponding to the input values. Arrays of shape `()` are converted to shape `(1,`` ``1,`` ``1)`, 1D arrays of shape `(N,)` are converted to shape `(1,`` ``N,`` ``1)`, 2D arrays of shape `(M,`` ``N)` are converted to shape `(M,`` ``N,`` ``1)`, and arrays of all other shapes are returned unchanged.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array") \| [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array")\]

See also

- [`jax.numpy.asarray()`](jax.numpy.asarray.html#jax.numpy.asarray "jax.numpy.asarray")

- [`jax.numpy.atleast_1d()`](jax.numpy.atleast_1d.html#jax.numpy.atleast_1d "jax.numpy.atleast_1d")

- [`jax.numpy.atleast_2d()`](jax.numpy.atleast_2d.html#jax.numpy.atleast_2d "jax.numpy.atleast_2d")

Examples

Scalar arguments are converted to 3D, size-1 arrays:

    >>> x = jnp.float32(1.0)
    >>> jnp.atleast_3d(x)
    Array([[[1.]]], dtype=float32)

1D arrays have a unit dimension prepended and appended:

    >>> y = jnp.arange(4)
    >>> jnp.atleast_3d(y).shape
    (1, 4, 1)

2D arrays have a unit dimension appended:

    >>> z = jnp.ones((2, 3))
    >>> jnp.atleast_3d(z).shape
    (2, 3, 1)

Multiple arguments can be passed to the function at once, in which case a list of results is returned:

    >>> x3, y3 = jnp.atleast_3d(x, y)
    >>> print(x3)
    [[[1.]]]
    >>> print(y3)
    [[[0]
      [1]
      [2]
      [3]]]

[](jax.numpy.atleast_2d.html "previous page")

previous

jax.numpy.atleast_2d

[](jax.numpy.average.html "next page")

next

jax.numpy.average

Contents

- [`atleast_3d()`](#jax.numpy.atleast_3d)

By The JAX authors

© Copyright 2024, The JAX Authors.\
