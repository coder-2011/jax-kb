- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.percentile

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.percentile.rst "Download source file")
-  .pdf

# jax.numpy.percentile

## Contents

- [`percentile()`](#jax.numpy.percentile)

# jax.numpy.percentile[\#](#jax-numpy-percentile "Link to this heading")

jax.numpy.percentile(*a*, *q*, *axis=None*, *out=None*, *overwrite_input=False*, *method='linear'*, *keepdims=False*, *\**, *weights=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/reductions.py#L2710-L2770)[\#](#jax.numpy.percentile "Link to this definition")  
Compute the percentile of the data along the specified axis.

JAX implementation of [`numpy.percentile()`](https://numpy.org/doc/stable/reference/generated/numpy.percentile.html#numpy.percentile "(in NumPy v2.4)").

Parameters:  
- **a** (*ArrayLike*) – N-dimensional array input.

- **q** (*ArrayLike*) – scalar or 1-dimensional array specifying the desired quantiles. `q` should contain integer or floating point values between `0` and `100`.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *...\]* *\|* *None*) – optional axis or tuple of axes along which to compute the quantile

- **out** (*None*) – not implemented by JAX; will error if not None

- **overwrite_input** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – not implemented by JAX; will error if not False

- **method** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – specify the interpolation method to use. Options are one of `["linear",`` ``"lower",`` ``"higher",`` ``"midpoint",`` ``"nearest"]`. default is `linear`.

- **keepdims** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – if True, then the returned array will have the same number of dimensions as the input. Default is False.

- **weights** (*ArrayLike* *\|* *None*) – keyword-only. optional array of weights for each element in a. Values with higher weights contribute more to the percentile calculation. The weights array must be broadcastable to the shape of a along the specified axis. Currently, weighted percentiles are only supported when method=”inverted_cdf”.

Returns:  
An array containing the specified percentiles along the specified axes.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.quantile()`](jax.numpy.quantile.html#jax.numpy.quantile "jax.numpy.quantile"): compute the quantile (0.0-1.0)

- [`jax.numpy.nanpercentile()`](jax.numpy.nanpercentile.html#jax.numpy.nanpercentile "jax.numpy.nanpercentile"): compute the percentile while ignoring NaNs

Examples

Computing the median and quartiles of a 1D array:

    >>> x = jnp.array([0, 1, 2, 3, 4, 5, 6])
    >>> q = jnp.array([25, 50, 75])
    >>> jnp.percentile(x, q)
    Array([1.5, 3. , 4.5], dtype=float32)

Computing the same percentiles with nearest rather than linear interpolation:

    >>> jnp.percentile(x, q, method='nearest')
    Array([1., 3., 4.], dtype=float32)

Computing weighted percentiles:

    >>> x = jnp.array([1, 2, 3, 4, 5])
    >>> weights = jnp.array([1, 1, 2, 1, 1])
    >>> jnp.percentile(x, 50, weights=weights, method='inverted_cdf')
    Array(3., dtype=float32)

[](jax.numpy.partition.html "previous page")

previous

jax.numpy.partition

[](jax.numpy.permute_dims.html "next page")

next

jax.numpy.permute_dims

Contents

- [`percentile()`](#jax.numpy.percentile)

By The JAX authors

© Copyright 2024, The JAX Authors.\
