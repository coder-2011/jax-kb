- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.nanmedian

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.nanmedian.rst "Download source file")
-  .pdf

# jax.numpy.nanmedian

## Contents

- [`nanmedian()`](#jax.numpy.nanmedian)

# jax.numpy.nanmedian[\#](#jax-numpy-nanmedian "Link to this heading")

jax.numpy.nanmedian(*a*, *axis=None*, *out=None*, *overwrite_input=False*, *keepdims=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/reductions.py#L2888-L2944)[\#](#jax.numpy.nanmedian "Link to this definition")  
Return the median of array elements along a given axis, ignoring NaNs.

JAX implementation of [`numpy.nanmedian()`](https://numpy.org/doc/stable/reference/generated/numpy.nanmedian.html#numpy.nanmedian "(in NumPy v2.4)").

Parameters:  
- **a** (*ArrayLike*) – input array.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *...\]* *\|* *None*) – optional, int or sequence of ints, default=None. Axis along which the median to be computed. If None, median is computed for the flattened array.

- **keepdims** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – bool, default=False. If true, reduced axes are left in the result with size 1.

- **out** (*None*) – Unused by JAX.

- **overwrite_input** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – Unused by JAX.

Returns:  
An array containing the median along the given axis, ignoring NaNs. If all elements along the given axis are NaNs, returns `nan`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.nanmean()`](jax.numpy.nanmean.html#jax.numpy.nanmean "jax.numpy.nanmean"): Compute the mean of array elements over a given axis, ignoring NaNs.

- [`jax.numpy.nanmax()`](jax.numpy.nanmax.html#jax.numpy.nanmax "jax.numpy.nanmax"): Compute the maximum of array elements over given axis, ignoring NaNs.

- [`jax.numpy.nanmin()`](jax.numpy.nanmin.html#jax.numpy.nanmin "jax.numpy.nanmin"): Compute the minimum of array elements over given axis, ignoring NaNs.

Examples

By default, the median is computed for the flattened array.

    >>> nan = jnp.nan
    >>> x = jnp.array([[2, nan, 7, nan],
    ...                [nan, 5, 9, 2],
    ...                [6, 1, nan, 3]])
    >>> jnp.nanmedian(x)
    Array(4., dtype=float32)

If `axis=1`, the median is computed along axis 1.

    >>> jnp.nanmedian(x, axis=1)
    Array([4.5, 5. , 3. ], dtype=float32)

If `keepdims=True`, `ndim` of the output is equal to that of the input.

    >>> jnp.nanmedian(x, axis=1, keepdims=True)
    Array([[4.5],
           [5. ],
           [3. ]], dtype=float32)

[](jax.numpy.nanmean.html "previous page")

previous

jax.numpy.nanmean

[](jax.numpy.nanmin.html "next page")

next

jax.numpy.nanmin

Contents

- [`nanmedian()`](#jax.numpy.nanmedian)

By The JAX authors

© Copyright 2024, The JAX Authors.\
