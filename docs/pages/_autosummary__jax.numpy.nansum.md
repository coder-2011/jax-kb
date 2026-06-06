- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.nansum

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.nansum.rst "Download source file")
-  .pdf

# jax.numpy.nansum

## Contents

- [`nansum()`](#jax.numpy.nansum)

# jax.numpy.nansum[\#](#jax-numpy-nansum "Link to this heading")

jax.numpy.nansum(*a*, *axis=None*, *dtype=None*, *out=None*, *keepdims=False*, *initial=None*, *where=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/reductions.py#L1564-L1648)[\#](#jax.numpy.nansum "Link to this definition")  
Return the sum of the array elements along a given axis, ignoring NaNs.

JAX implementation of [`numpy.nansum()`](https://numpy.org/doc/stable/reference/generated/numpy.nansum.html#numpy.nansum "(in NumPy v2.4)").

Parameters:  
- **a** (*ArrayLike*) – Input array.

- **axis** (*Axis*) – int or sequence of ints, default=None. Axis along which the sum is computed. If None, the sum is computed along the flattened array.

- **dtype** (*DTypeLike* *\|* *None*) – The type of the output array. Default=None.

- **keepdims** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – bool, default=False. If True, reduced axes are left in the result with size 1.

- **initial** (*ArrayLike* *\|* *None*) – int or array, default=None. Initial value for the sum.

- **where** (*ArrayLike* *\|* *None*) – array of boolean dtype, default=None. The elements to be used in the sum. Array should be broadcast compatible to the input.

- **out** (*None*) – Unused by JAX.

Returns:  
An array containing the sum of array elements along the given axis, ignoring NaNs. If all elements along the given axis are NaNs, returns 0.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.nanmin()`](jax.numpy.nanmin.html#jax.numpy.nanmin "jax.numpy.nanmin"): Compute the minimum of array elements along a given axis, ignoring NaNs.

- [`jax.numpy.nanmax()`](jax.numpy.nanmax.html#jax.numpy.nanmax "jax.numpy.nanmax"): Compute the maximum of array elements along a given axis, ignoring NaNs.

- [`jax.numpy.nanprod()`](jax.numpy.nanprod.html#jax.numpy.nanprod "jax.numpy.nanprod"): Compute the product of array elements along a given axis, ignoring NaNs.

- [`jax.numpy.nanmean()`](jax.numpy.nanmean.html#jax.numpy.nanmean "jax.numpy.nanmean"): Compute the mean of array elements along a given axis, ignoring NaNs.

Examples

By default, `jnp.nansum` computes the sum of elements along the flattened array.

    >>> nan = jnp.nan
    >>> x = jnp.array([[3, nan, 4, 5],
    ...                [nan, -2, nan, 7],
    ...                [2, 1, 6, nan]])
    >>> jnp.nansum(x)
    Array(26., dtype=float32)

If `axis=1`, the sum will be computed along axis 1.

    >>> jnp.nansum(x, axis=1)
    Array([12.,  5.,  9.], dtype=float32)

If `keepdims=True`, `ndim` of the output will be same of that of the input.

    >>> jnp.nansum(x, axis=1, keepdims=True)
    Array([[12.],
           [ 5.],
           [ 9.]], dtype=float32)

To include only specific elements in computing the sum, you can use `where`.

    >>> where=jnp.array([[1, 0, 1, 0],
    ...                  [0, 0, 1, 1],
    ...                  [1, 1, 1, 0]], dtype=bool)
    >>> jnp.nansum(x, axis=1, keepdims=True, where=where)
    Array([[7.],
           [7.],
           [9.]], dtype=float32)

If `where` is `False` at all elements, `jnp.nansum` returns 0 along the given axis.

    >>> where = jnp.array([[False],
    ...                    [False],
    ...                    [False]])
    >>> jnp.nansum(x, axis=0, keepdims=True, where=where)
    Array([[0., 0., 0., 0.]], dtype=float32)

[](jax.numpy.nanstd.html "previous page")

previous

jax.numpy.nanstd

[](jax.numpy.nanvar.html "next page")

next

jax.numpy.nanvar

Contents

- [`nansum()`](#jax.numpy.nansum)

By The JAX authors

© Copyright 2024, The JAX Authors.\
