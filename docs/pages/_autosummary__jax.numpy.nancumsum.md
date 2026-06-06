- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.nancumsum

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.nancumsum.rst "Download source file")
-  .pdf

# jax.numpy.nancumsum

## Contents

- [`nancumsum()`](#jax.numpy.nancumsum)

# jax.numpy.nancumsum[\#](#jax-numpy-nancumsum "Link to this heading")

jax.numpy.nancumsum(*a*, *axis=None*, *dtype=None*, *out=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/reductions.py#L2152-L2200)[\#](#jax.numpy.nancumsum "Link to this definition")  
Cumulative sum of elements along an axis, ignoring NaN values.

JAX implementation of [`numpy.nancumsum()`](https://numpy.org/doc/stable/reference/generated/numpy.nancumsum.html#numpy.nancumsum "(in NumPy v2.4)").

Parameters:  
- **a** (*ArrayLike*) – N-dimensional array to be accumulated.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – integer axis along which to accumulate. If None (default), then array will be flattened and accumulated along the flattened axis.

- **dtype** (*DTypeLike* *\|* *None*) – optionally specify the dtype of the output. If not specified, then the output dtype will match the input dtype.

- **out** (*None*) – unused by JAX

Returns:  
An array containing the accumulated sum along the given axis.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.cumsum()`](jax.numpy.cumsum.html#jax.numpy.cumsum "jax.numpy.cumsum"): cumulative sum without ignoring NaN values.

- [`jax.numpy.cumulative_sum()`](jax.numpy.cumulative_sum.html#jax.numpy.cumulative_sum "jax.numpy.cumulative_sum"): cumulative sum via the array API standard.

- `jax.numpy.add.accumulate()`: cumulative sum via ufunc methods.

- [`jax.numpy.sum()`](jax.numpy.sum.html#jax.numpy.sum "jax.numpy.sum"): sum along axis

Examples

    >>> x = jnp.array([[1., 2., jnp.nan],
    ...                [4., jnp.nan, 6.]])

The standard cumulative sum will propagate NaN values:

    >>> jnp.cumsum(x)
    Array([ 1.,  3., nan, nan, nan, nan], dtype=float32)

[`nancumsum()`](#jax.numpy.nancumsum "jax.numpy.nancumsum") will ignore NaN values, effectively replacing them with zeros:

    >>> jnp.nancumsum(x)
    Array([ 1.,  3.,  3.,  7.,  7., 13.], dtype=float32)

Cumulative sum along axis 1:

    >>> jnp.nancumsum(x, axis=1)
    Array([[ 1.,  3.,  3.],
           [ 4.,  4., 10.]], dtype=float32)

[](jax.numpy.nancumprod.html "previous page")

previous

jax.numpy.nancumprod

[](jax.numpy.nanmax.html "next page")

next

jax.numpy.nanmax

Contents

- [`nancumsum()`](#jax.numpy.nancumsum)

By The JAX authors

© Copyright 2024, The JAX Authors.\
