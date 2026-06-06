- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.atleast_2d

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.atleast_2d.rst "Download source file")
-  .pdf

# jax.numpy.atleast_2d

## Contents

- [`atleast_2d()`](#jax.numpy.atleast_2d)

# jax.numpy.atleast_2d[\#](#jax-numpy-atleast-2d "Link to this heading")

jax.numpy.atleast_2d(*\*arys*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L5127-L5179)[\#](#jax.numpy.atleast_2d "Link to this definition")  
Convert inputs to arrays with at least 2 dimensions.

JAX implementation of [`numpy.atleast_2d()`](https://numpy.org/doc/stable/reference/generated/numpy.atleast_2d.html#numpy.atleast_2d "(in NumPy v2.4)").

Parameters:  
- **arguments.** (*zero* *or* *more arraylike*)

- **arys** (*ArrayLike*)

Returns:  
an array or list of arrays corresponding to the input values. Arrays of shape `()` are converted to shape `(1,`` ``1)`, 1D arrays of shape `(N,)` are converted to shape `(1,`` ``N)`, and arrays of all other shapes are returned unchanged.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array") \| [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array")\]

See also

- [`jax.numpy.asarray()`](jax.numpy.asarray.html#jax.numpy.asarray "jax.numpy.asarray")

- [`jax.numpy.atleast_1d()`](jax.numpy.atleast_1d.html#jax.numpy.atleast_1d "jax.numpy.atleast_1d")

- [`jax.numpy.atleast_3d()`](jax.numpy.atleast_3d.html#jax.numpy.atleast_3d "jax.numpy.atleast_3d")

Examples

Scalar arguments are converted to 2D, size-1 arrays:

    >>> x = jnp.float32(1.0)
    >>> jnp.atleast_2d(x)
    Array([[1.]], dtype=float32)

One-dimensional arguments have a unit dimension prepended to the shape:

    >>> y = jnp.arange(4)
    >>> jnp.atleast_2d(y)
    Array([[0, 1, 2, 3]], dtype=int32)

Higher dimensional inputs are returned unchanged:

    >>> z = jnp.ones((2, 3))
    >>> jnp.atleast_2d(z)
    Array([[1., 1., 1.],
           [1., 1., 1.]], dtype=float32)

Multiple arguments can be passed to the function at once, in which case a list of results is returned:

    >>> jnp.atleast_2d(x, y)
    [Array([[1.]], dtype=float32), Array([[0, 1, 2, 3]], dtype=int32)]

[](jax.numpy.atleast_1d.html "previous page")

previous

jax.numpy.atleast_1d

[](jax.numpy.atleast_3d.html "next page")

next

jax.numpy.atleast_3d

Contents

- [`atleast_2d()`](#jax.numpy.atleast_2d)

By The JAX authors

© Copyright 2024, The JAX Authors.\
