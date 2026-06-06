- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.nanmean

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.nanmean.rst "Download source file")
-  .pdf

# jax.numpy.nanmean

## Contents

- [`nanmean()`](#jax.numpy.nanmean)

# jax.numpy.nanmean[\#](#jax-numpy-nanmean "Link to this heading")

jax.numpy.nanmean(*a*, *axis=None*, *dtype=None*, *out=None*, *keepdims=False*, *where=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/reductions.py#L1735-L1825)[\#](#jax.numpy.nanmean "Link to this definition")  
Return the mean of the array elements along a given axis, ignoring NaNs.

JAX implementation of [`numpy.nanmean()`](https://numpy.org/doc/stable/reference/generated/numpy.nanmean.html#numpy.nanmean "(in NumPy v2.4)").

Parameters:  
- **a** (*ArrayLike*) – Input array.

- **axis** (*Axis*) – int or sequence of ints, default=None. Axis along which the mean is computed. If None, the mean is computed along the flattened array.

- **dtype** (*DTypeLike* *\|* *None*) – The type of the output array. Default=None.

- **keepdims** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – bool, default=False. If True, reduced axes are left in the result with size 1.

- **where** (*ArrayLike* *\|* *None*) – array of boolean dtype, default=None. The elements to be used in computing mean. Array should be broadcast compatible to the input.

- **out** (*None*) – Unused by JAX.

Returns:  
An array containing the mean of array elements along the given axis, ignoring NaNs. If all elements along the given axis are NaNs, returns `nan`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.nanmin()`](jax.numpy.nanmin.html#jax.numpy.nanmin "jax.numpy.nanmin"): Compute the minimum of array elements along a given axis, ignoring NaNs.

- [`jax.numpy.nanmax()`](jax.numpy.nanmax.html#jax.numpy.nanmax "jax.numpy.nanmax"): Compute the maximum of array elements along a given axis, ignoring NaNs.

- [`jax.numpy.nansum()`](jax.numpy.nansum.html#jax.numpy.nansum "jax.numpy.nansum"): Compute the sum of array elements along a given axis, ignoring NaNs.

- [`jax.numpy.nanprod()`](jax.numpy.nanprod.html#jax.numpy.nanprod "jax.numpy.nanprod"): Compute the product of array elements along a given axis, ignoring NaNs.

Examples

By default, `jnp.nanmean` computes the mean of elements along the flattened array.

    >>> nan = jnp.nan
    >>> x = jnp.array([[2, nan, 4, 3],
    ...                [nan, -2, nan, 9],
    ...                [4, -7, 6, nan]])
    >>> jnp.nanmean(x)
    Array(2.375, dtype=float32)

If `axis=1`, mean will be computed along axis 1.

    >>> jnp.nanmean(x, axis=1)
    Array([3. , 3.5, 1. ], dtype=float32)

If `keepdims=True`, `ndim` of the output will be same of that of the input.

    >>> jnp.nanmean(x, axis=1, keepdims=True)
    Array([[3. ],
           [3.5],
           [1. ]], dtype=float32)

`where` can be used to include only specific elements in computing the mean.

    >>> where = jnp.array([[1, 0, 1, 0],
    ...                    [0, 0, 1, 1],
    ...                    [1, 1, 0, 1]], dtype=bool)
    >>> jnp.nanmean(x, axis=1, keepdims=True, where=where)
    Array([[ 3. ],
           [ 9. ],
           [-1.5]], dtype=float32)

If `where` is `False` at all elements, `jnp.nanmean` returns `nan` along the given axis.

    >>> where = jnp.array([[False],
    ...                    [False],
    ...                    [False]])
    >>> jnp.nanmean(x, axis=0, keepdims=True, where=where)
    Array([[nan, nan, nan, nan]], dtype=float32)

[](jax.numpy.nanmax.html "previous page")

previous

jax.numpy.nanmax

[](jax.numpy.nanmedian.html "next page")

next

jax.numpy.nanmedian

Contents

- [`nanmean()`](#jax.numpy.nanmean)

By The JAX authors

© Copyright 2024, The JAX Authors.\
