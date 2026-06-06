- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.nanprod

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.nanprod.rst "Download source file")
-  .pdf

# jax.numpy.nanprod

## Contents

- [`nanprod()`](#jax.numpy.nanprod)

# jax.numpy.nanprod[\#](#jax-numpy-nanprod "Link to this heading")

jax.numpy.nanprod(*a*, *axis=None*, *dtype=None*, *out=None*, *keepdims=False*, *initial=None*, *where=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/reductions.py#L1650-L1733)[\#](#jax.numpy.nanprod "Link to this definition")  
Return the product of the array elements along a given axis, ignoring NaNs.

JAX implementation of [`numpy.nanprod()`](https://numpy.org/doc/stable/reference/generated/numpy.nanprod.html#numpy.nanprod "(in NumPy v2.4)").

Parameters:  
- **a** (*ArrayLike*) – Input array.

- **axis** (*Axis*) – int or sequence of ints, default=None. Axis along which the product is computed. If None, the product is computed along the flattened array.

- **dtype** (*DTypeLike* *\|* *None*) – The type of the output array. Default=None.

- **keepdims** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – bool, default=False. If True, reduced axes are left in the result with size 1.

- **initial** (*ArrayLike* *\|* *None*) – int or array, default=None. Initial value for the product.

- **where** (*ArrayLike* *\|* *None*) – array of boolean dtype, default=None. The elements to be used in the product. Array should be broadcast compatible to the input.

- **out** (*None*) – Unused by JAX.

Returns:  
An array containing the product of array elements along the given axis, ignoring NaNs. If all elements along the given axis are NaNs, returns 1.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.nanmin()`](jax.numpy.nanmin.html#jax.numpy.nanmin "jax.numpy.nanmin"): Compute the minimum of array elements along a given axis, ignoring NaNs.

- [`jax.numpy.nanmax()`](jax.numpy.nanmax.html#jax.numpy.nanmax "jax.numpy.nanmax"): Compute the maximum of array elements along a given axis, ignoring NaNs.

- [`jax.numpy.nansum()`](jax.numpy.nansum.html#jax.numpy.nansum "jax.numpy.nansum"): Compute the sum of array elements along a given axis, ignoring NaNs.

- [`jax.numpy.nanmean()`](jax.numpy.nanmean.html#jax.numpy.nanmean "jax.numpy.nanmean"): Compute the mean of array elements along a given axis, ignoring NaNs.

Examples

By default, `jnp.nanprod` computes the product of elements along the flattened array.

    >>> nan = jnp.nan
    >>> x = jnp.array([[nan, 3, 4, nan],
    ...                [5, nan, 1, 3],
    ...                [2, 1, nan, 1]])
    >>> jnp.nanprod(x)
    Array(360., dtype=float32)

If `axis=1`, the product will be computed along axis 1.

    >>> jnp.nanprod(x, axis=1)
    Array([12., 15.,  2.], dtype=float32)

If `keepdims=True`, `ndim` of the output will be same of that of the input.

    >>> jnp.nanprod(x, axis=1, keepdims=True)
    Array([[12.],
           [15.],
           [ 2.]], dtype=float32)

To include only specific elements in computing the maximum, you can use `where`.

    >>> where=jnp.array([[1, 0, 1, 0],
    ...                  [0, 0, 1, 1],
    ...                  [1, 1, 1, 0]], dtype=bool)
    >>> jnp.nanprod(x, axis=1, keepdims=True, where=where)
    Array([[4.],
           [3.],
           [2.]], dtype=float32)

If `where` is `False` at all elements, `jnp.nanprod` returns 1 along the given axis.

    >>> where = jnp.array([[False],
    ...                    [False],
    ...                    [False]])
    >>> jnp.nanprod(x, axis=0, keepdims=True, where=where)
    Array([[1., 1., 1., 1.]], dtype=float32)

[](jax.numpy.nanpercentile.html "previous page")

previous

jax.numpy.nanpercentile

[](jax.numpy.nanquantile.html "next page")

next

jax.numpy.nanquantile

Contents

- [`nanprod()`](#jax.numpy.nanprod)

By The JAX authors

© Copyright 2024, The JAX Authors.\
