- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.average

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.average.rst "Download source file")
-  .pdf

# jax.numpy.average

## Contents

- [`average()`](#jax.numpy.average)

# jax.numpy.average[\#](#jax-numpy-average "Link to this heading")

jax.numpy.average(*a*, *axis=None*, *weights=None*, *returned=False*, *keepdims=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/reductions.py#L936-L992)[\#](#jax.numpy.average "Link to this definition")  
Compute the weighed average.

JAX Implementation of [`numpy.average()`](https://numpy.org/doc/stable/reference/generated/numpy.average.html#numpy.average "(in NumPy v2.4)").

Parameters:  
- **a** (*ArrayLike*) – array to be averaged

- **axis** (*Axis*) – an optional integer or sequence of integers specifying the axis along which the mean to be computed. If not specified, mean is computed along all the axes.

- **weights** (*ArrayLike* *\|* *None*) – an optional array of weights for a weighted average. This must either exactly match the shape of a, or if axis is specified, it must have shape `a.shape[axis]` for a single axis, or shape `tuple(a.shape[ax]`` ``for`` ``ax`` ``in`` ``axis)` for multiple axes.

- **returned** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – If False (default) then return only the average. If True then return both the average and the normalization factor (i.e. the sum of weights).

- **keepdims** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – If True, reduced axes are left in the result with size 1. If False (default) then reduced axes are squeezed out.

Returns:  
An array `average` or tuple of arrays `(average,`` ``normalization)` if `returned` is True.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array") \| [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\]

See also

- [`jax.numpy.mean()`](jax.numpy.mean.html#jax.numpy.mean "jax.numpy.mean"): unweighted mean.

Examples

Simple average:

    >>> x = jnp.array([1, 2, 3, 2, 4])
    >>> jnp.average(x)
    Array(2.4, dtype=float32)

Weighted average:

    >>> weights = jnp.array([2, 1, 3, 2, 2])
    >>> jnp.average(x, weights=weights)
    Array(2.5, dtype=float32)

Use `returned=True` to optionally return the normalization, i.e. the sum of weights:

    >>> jnp.average(x, returned=True)
    (Array(2.4, dtype=float32), Array(5., dtype=float32))
    >>> jnp.average(x, weights=weights, returned=True)
    (Array(2.5, dtype=float32), Array(10., dtype=float32))

Weighted average along a specified axis:

    >>> x = jnp.array([[8, 2, 7],
    ...                [3, 6, 4]])
    >>> weights = jnp.array([1, 2, 3])
    >>> jnp.average(x, weights=weights, axis=1)
    Array([5.5, 4.5], dtype=float32)

[](jax.numpy.atleast_3d.html "previous page")

previous

jax.numpy.atleast_3d

[](jax.numpy.bartlett.html "next page")

next

jax.numpy.bartlett

Contents

- [`average()`](#jax.numpy.average)

By The JAX authors

© Copyright 2024, The JAX Authors.\
