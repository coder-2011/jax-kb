- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.quantile

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.quantile.rst "Download source file")
-  .pdf

# jax.numpy.quantile

## Contents

- [`quantile()`](#jax.numpy.quantile)

# jax.numpy.quantile[\#](#jax-numpy-quantile "Link to this heading")

jax.numpy.quantile(*a*, *q*, *axis=None*, *out=None*, *overwrite_input=False*, *method='linear'*, *keepdims=False*, *\**, *weights=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/reductions.py#L2384-L2444)[\#](#jax.numpy.quantile "Link to this definition")  
Compute the quantile of the data along the specified axis.

JAX implementation of [`numpy.quantile()`](https://numpy.org/doc/stable/reference/generated/numpy.quantile.html#numpy.quantile "(in NumPy v2.4)").

Parameters:  
- **a** (*ArrayLike*) – N-dimensional array input.

- **q** (*ArrayLike*) – scalar or 1-dimensional array specifying the desired quantiles. `q` should contain floating-point values between `0.0` and `1.0`.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *...\]* *\|* *None*) – optional axis or tuple of axes along which to compute the quantile

- **out** (*None*) – not implemented by JAX; will error if not None

- **overwrite_input** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – not implemented by JAX; will error if not False

- **method** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – specify the interpolation method to use. Options are one of `["linear",`` ``"lower",`` ``"higher",`` ``"midpoint",`` ``"nearest"]`. default is `linear`.

- **keepdims** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – if True, then the returned array will have the same number of dimensions as the input. Default is False.

- **weights** (*ArrayLike* *\|* *None*) – keyword-only. optional array of weights associated with the values in `a`. Each value in `a` contributes to the quantile according to its associated weight. The weights array must be broadcastable to the same shape as `a`. Only works with `method="inverted_cdf"`.

Returns:  
An array containing the specified quantiles along the specified axes.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.nanquantile()`](jax.numpy.nanquantile.html#jax.numpy.nanquantile "jax.numpy.nanquantile"): compute the quantile while ignoring NaNs

- [`jax.numpy.percentile()`](jax.numpy.percentile.html#jax.numpy.percentile "jax.numpy.percentile"): compute the percentile (0-100)

Examples

Computing the median and quartiles of an array, with linear interpolation:

    >>> x = jnp.arange(10)
    >>> q = jnp.array([0.25, 0.5, 0.75])
    >>> jnp.quantile(x, q)
    Array([2.25, 4.5 , 6.75], dtype=float32)

Computing the quartiles using nearest-value interpolation:

    >>> jnp.quantile(x, q, method='nearest')
    Array([2., 4., 7.], dtype=float32)

Computing weighted quantiles:

    >>> x = jnp.array([1, 2, 3, 4, 5])
    >>> weights = jnp.array([1, 1, 2, 1, 1])
    >>> jnp.quantile(x, 0.5, weights=weights, method='inverted_cdf')
    Array(3., dtype=float32)

[](jax.numpy.put_along_axis.html "previous page")

previous

jax.numpy.put_along_axis

[](jax.numpy.r_.html "next page")

next

jax.numpy.r\_

Contents

- [`quantile()`](#jax.numpy.quantile)

By The JAX authors

© Copyright 2024, The JAX Authors.\
