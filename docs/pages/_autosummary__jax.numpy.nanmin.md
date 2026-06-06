- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.nanmin

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.nanmin.rst "Download source file")
-  .pdf

# jax.numpy.nanmin

## Contents

- [`nanmin()`](#jax.numpy.nanmin)

# jax.numpy.nanmin[\#](#jax-numpy-nanmin "Link to this heading")

jax.numpy.nanmin(*a*, *axis=None*, *out=None*, *keepdims=False*, *initial=None*, *where=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/reductions.py#L1398-L1479)[\#](#jax.numpy.nanmin "Link to this definition")  
Return the minimum of the array elements along a given axis, ignoring NaNs.

JAX implementation of [`numpy.nanmin()`](https://numpy.org/doc/stable/reference/generated/numpy.nanmin.html#numpy.nanmin "(in NumPy v2.4)").

Parameters:  
- **a** (*ArrayLike*) – Input array.

- **axis** (*Axis*) – int or sequence of ints, default=None. Axis along which the minimum is computed. If None, the minimum is computed along the flattened array.

- **keepdims** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – bool, default=False. If True, reduced axes are left in the result with size 1.

- **initial** (*ArrayLike* *\|* *None*) – int or array, default=None. Initial value for the minimum.

- **where** (*ArrayLike* *\|* *None*) – array of boolean dtype, default=None. The elements to be used in the minimum. Array should be broadcast compatible to the input. `initial` must be specified when `where` is used.

- **out** (*None*) – Unused by JAX.

Returns:  
An array of minimum values along the given axis, ignoring NaNs. If all values are NaNs along the given axis, returns `nan`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.nanmax()`](jax.numpy.nanmax.html#jax.numpy.nanmax "jax.numpy.nanmax"): Compute the maximum of array elements along a given axis, ignoring NaNs.

- [`jax.numpy.nansum()`](jax.numpy.nansum.html#jax.numpy.nansum "jax.numpy.nansum"): Compute the sum of array elements along a given axis, ignoring NaNs.

- [`jax.numpy.nanprod()`](jax.numpy.nanprod.html#jax.numpy.nanprod "jax.numpy.nanprod"): Compute the product of array elements along a given axis, ignoring NaNs.

- [`jax.numpy.nanmean()`](jax.numpy.nanmean.html#jax.numpy.nanmean "jax.numpy.nanmean"): Compute the mean of array elements along a given axis, ignoring NaNs.

Examples

By default, `jnp.nanmin` computes the minimum of elements along the flattened array.

    >>> nan = jnp.nan
    >>> x = jnp.array([[1, nan, 4, 5],
    ...                [nan, -2, nan, -4],
    ...                [2, 1, 3, nan]])
    >>> jnp.nanmin(x)
    Array(-4., dtype=float32)

If `axis=1`, the maximum will be computed along axis 1.

    >>> jnp.nanmin(x, axis=1)
    Array([ 1., -4.,  1.], dtype=float32)

If `keepdims=True`, `ndim` of the output will be same of that of the input.

    >>> jnp.nanmin(x, axis=1, keepdims=True)
    Array([[ 1.],
           [-4.],
           [ 1.]], dtype=float32)

To include only specific elements in computing the maximum, you can use `where`. It can either have same dimension as input

    >>> where=jnp.array([[0, 0, 1, 0],
    ...                  [0, 0, 1, 1],
    ...                  [1, 1, 1, 0]], dtype=bool)
    >>> jnp.nanmin(x, axis=1, keepdims=True, initial=0, where=where)
    Array([[ 0.],
           [-4.],
           [ 0.]], dtype=float32)

or must be broadcast compatible with input.

    >>> where = jnp.array([[False],
    ...                    [True],
    ...                    [False]])
    >>> jnp.nanmin(x, axis=0, keepdims=True, initial=0, where=where)
    Array([[ 0., -2.,  0., -4.]], dtype=float32)

[](jax.numpy.nanmedian.html "previous page")

previous

jax.numpy.nanmedian

[](jax.numpy.nanpercentile.html "next page")

next

jax.numpy.nanpercentile

Contents

- [`nanmin()`](#jax.numpy.nanmin)

By The JAX authors

© Copyright 2024, The JAX Authors.\
