- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.nanmax

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.nanmax.rst "Download source file")
-  .pdf

# jax.numpy.nanmax

## Contents

- [`nanmax()`](#jax.numpy.nanmax)

# jax.numpy.nanmax[\#](#jax-numpy-nanmax "Link to this heading")

jax.numpy.nanmax(*a*, *axis=None*, *out=None*, *keepdims=False*, *initial=None*, *where=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/reductions.py#L1481-L1562)[\#](#jax.numpy.nanmax "Link to this definition")  
Return the maximum of the array elements along a given axis, ignoring NaNs.

JAX implementation of [`numpy.nanmax()`](https://numpy.org/doc/stable/reference/generated/numpy.nanmax.html#numpy.nanmax "(in NumPy v2.4)").

Parameters:  
- **a** (*ArrayLike*) – Input array.

- **axis** (*Axis*) – int or sequence of ints, default=None. Axis along which the maximum is computed. If None, the maximum is computed along the flattened array.

- **keepdims** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – bool, default=False. If True, reduced axes are left in the result with size 1.

- **initial** (*ArrayLike* *\|* *None*) – int or array, default=None. Initial value for the maximum.

- **where** (*ArrayLike* *\|* *None*) – array of boolean dtype, default=None. The elements to be used in the maximum. Array should be broadcast compatible to the input. `initial` must be specified when `where` is used.

- **out** (*None*) – Unused by JAX.

Returns:  
An array of maximum values along the given axis, ignoring NaNs. If all values are NaNs along the given axis, returns `nan`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.nanmin()`](jax.numpy.nanmin.html#jax.numpy.nanmin "jax.numpy.nanmin"): Compute the minimum of array elements along a given axis, ignoring NaNs.

- [`jax.numpy.nansum()`](jax.numpy.nansum.html#jax.numpy.nansum "jax.numpy.nansum"): Compute the sum of array elements along a given axis, ignoring NaNs.

- [`jax.numpy.nanprod()`](jax.numpy.nanprod.html#jax.numpy.nanprod "jax.numpy.nanprod"): Compute the product of array elements along a given axis, ignoring NaNs.

- [`jax.numpy.nanmean()`](jax.numpy.nanmean.html#jax.numpy.nanmean "jax.numpy.nanmean"): Compute the mean of array elements along a given axis, ignoring NaNs.

Examples

By default, `jnp.nanmax` computes the maximum of elements along the flattened array.

    >>> nan = jnp.nan
    >>> x = jnp.array([[8, nan, 4, 6],
    ...                [nan, -2, nan, -4],
    ...                [-2, 1, 7, nan]])
    >>> jnp.nanmax(x)
    Array(8., dtype=float32)

If `axis=1`, the maximum will be computed along axis 1.

    >>> jnp.nanmax(x, axis=1)
    Array([ 8., -2.,  7.], dtype=float32)

If `keepdims=True`, `ndim` of the output will be same of that of the input.

    >>> jnp.nanmax(x, axis=1, keepdims=True)
    Array([[ 8.],
           [-2.],
           [ 7.]], dtype=float32)

To include only specific elements in computing the maximum, you can use `where`. It can either have same dimension as input

    >>> where=jnp.array([[0, 0, 1, 0],
    ...                  [0, 0, 1, 1],
    ...                  [1, 1, 1, 0]], dtype=bool)
    >>> jnp.nanmax(x, axis=1, keepdims=True, initial=0, where=where)
    Array([[4.],
           [0.],
           [7.]], dtype=float32)

or must be broadcast compatible with input.

    >>> where = jnp.array([[True],
    ...                    [False],
    ...                    [False]])
    >>> jnp.nanmax(x, axis=0, keepdims=True, initial=0, where=where)
    Array([[8., 0., 4., 6.]], dtype=float32)

[](jax.numpy.nancumsum.html "previous page")

previous

jax.numpy.nancumsum

[](jax.numpy.nanmean.html "next page")

next

jax.numpy.nanmean

Contents

- [`nanmax()`](#jax.numpy.nanmax)

By The JAX authors

© Copyright 2024, The JAX Authors.\
