- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.nanquantile

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.nanquantile.rst "Download source file")
-  .pdf

# jax.numpy.nanquantile

## Contents

- [`nanquantile()`](#jax.numpy.nanquantile)

# jax.numpy.nanquantile[\#](#jax-numpy-nanquantile "Link to this heading")

jax.numpy.nanquantile(*a*, *q*, *axis=None*, *out=None*, *overwrite_input=False*, *method='linear'*, *keepdims=False*, *\**, *weights=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/reductions.py#L2446-L2509)[\#](#jax.numpy.nanquantile "Link to this definition")  
Compute the quantile of the data along the specified axis, ignoring NaNs.

JAX implementation of [`numpy.nanquantile()`](https://numpy.org/doc/stable/reference/generated/numpy.nanquantile.html#numpy.nanquantile "(in NumPy v2.4)").

Parameters:  
- **a** (*ArrayLike*) – N-dimensional array input.

- **q** (*ArrayLike*) – scalar or 1-dimensional array specifying the desired quantiles. `q` should contain floating-point values between `0.0` and `1.0`.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *...\]* *\|* *None*) – optional axis or tuple of axes along which to compute the quantile

- **out** (*None*) – not implemented by JAX; will error if not None

- **overwrite_input** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – not implemented by JAX; will error if not False

- **method** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – specify the interpolation method to use. Options are one of `["linear",`` ``"lower",`` ``"higher",`` ``"midpoint",`` ``"nearest"]`. default is `linear`.

- **keepdims** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – if True, then the returned array will have the same number of dimensions as the input. Default is False.

- **weights** (*ArrayLike* *\|* *None*) – keyword-only. Optional array of weights for each element in a. Values with higher weights contribute more to the quantile calculation. The weights array must be broadcastable to the shape of a along the specified axis. NaN values in a are ignored when computing the quantiles. Weighted quantiles are currently only supported when method=”inverted_cdf”.

Returns:  
An array containing the specified quantiles along the specified axes.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.quantile()`](jax.numpy.quantile.html#jax.numpy.quantile "jax.numpy.quantile"): compute the quantile without ignoring nans

- [`jax.numpy.nanpercentile()`](jax.numpy.nanpercentile.html#jax.numpy.nanpercentile "jax.numpy.nanpercentile"): compute the percentile (0-100)

Examples

Computing the median and quartiles of a 1D array:

    >>> x = jnp.array([0, 1, 2, jnp.nan, 3, 4, 5, 6])
    >>> q = jnp.array([0.25, 0.5, 0.75])

Because of the NaN value, [`jax.numpy.quantile()`](jax.numpy.quantile.html#jax.numpy.quantile "jax.numpy.quantile") returns all NaNs, while [`nanquantile()`](#jax.numpy.nanquantile "jax.numpy.nanquantile") ignores them:

    >>> jnp.quantile(x, q)
    Array([nan, nan, nan], dtype=float32)
    >>> jnp.nanquantile(x, q)
    Array([1.5, 3. , 4.5], dtype=float32)

Computing weighted quantiles while ignoring NaNs:

    >>> x = jnp.array([1, 2, jnp.nan, 4, 5])
    >>> weights = jnp.array([1, 1, 1, 2, 1])
    >>> jnp.nanquantile(x, 0.5, weights=weights, method='inverted_cdf')
    Array(4.0, dtype=float32)

[](jax.numpy.nanprod.html "previous page")

previous

jax.numpy.nanprod

[](jax.numpy.nanstd.html "next page")

next

jax.numpy.nanstd

Contents

- [`nanquantile()`](#jax.numpy.nanquantile)

By The JAX authors

© Copyright 2024, The JAX Authors.\
