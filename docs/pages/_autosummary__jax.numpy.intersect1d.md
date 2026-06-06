- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.intersect1d

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.intersect1d.rst "Download source file")
-  .pdf

# jax.numpy.intersect1d

## Contents

- [`intersect1d()`](#jax.numpy.intersect1d)

# jax.numpy.intersect1d[\#](#jax-numpy-intersect1d "Link to this heading")

jax.numpy.intersect1d(*ar1*, *ar2*, *assume_unique=False*, *return_indices=False*, *\**, *size=None*, *fill_value=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/setops.py#L429-L536)[\#](#jax.numpy.intersect1d "Link to this definition")  
Compute the set intersection of two 1D arrays.

JAX implementation of [`numpy.intersect1d()`](https://numpy.org/doc/stable/reference/generated/numpy.intersect1d.html#numpy.intersect1d "(in NumPy v2.4)").

Because the size of the output of `intersect1d` is data-dependent, the function is not typically compatible with [`jit()`](jax.jit.html#jax.jit "jax.jit") and other JAX transformations. The JAX version adds the optional `size` argument which must be specified statically for `jnp.intersect1d` to be used in such contexts.

Parameters:  
- **ar1** (*ArrayLike*) – first array of values to intersect.

- **ar2** (*ArrayLike*) – second array of values to intersect.

- **assume_unique** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – if True, assume the input arrays contain unique values. This allows a more efficient implementation, but if `assume_unique` is True and the input arrays contain duplicates, the behavior is undefined. default: False.

- **return_indices** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – If True, return arrays of indices specifying where the intersected values first appear in the input arrays.

- **size** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – if specified, return only the first `size` sorted elements. If there are fewer elements than `size` indicates, the return value will be padded with `fill_value`, and returned indices will be padded with an out-of-bound index.

- **fill_value** (*ArrayLike* *\|* *None*) – when `size` is specified and there are fewer than the indicated number of elements, fill the remaining entries `fill_value`. Defaults to the smallest value in the intersection.

Returns:  
An array `intersection`, or if `return_indices=True`, a tuple of arrays `(intersection,`` ``ar1_indices,`` ``ar2_indices)`. Returned values are

- `intersection`: A 1D array containing each value that appears in both `ar1` and `ar2`.

- `ar1_indices`: *(returned if return_indices=True)* an array of shape `intersection.shape` containing the indices in flattened `ar1` of values in `intersection`. For 1D inputs, `intersection` is equivalent to `ar1[ar1_indices]`.

- `ar2_indices`: *(returned if return_indices=True)* an array of shape `intersection.shape` containing the indices in flattened `ar2` of values in `intersection`. For 1D inputs, `intersection` is equivalent to `ar2[ar2_indices]`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array") \| [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\]

See also

- [`jax.numpy.union1d()`](jax.numpy.union1d.html#jax.numpy.union1d "jax.numpy.union1d"): the set union of two 1D arrays.

- [`jax.numpy.setxor1d()`](jax.numpy.setxor1d.html#jax.numpy.setxor1d "jax.numpy.setxor1d"): the set XOR of two 1D arrays.

- [`jax.numpy.setdiff1d()`](jax.numpy.setdiff1d.html#jax.numpy.setdiff1d "jax.numpy.setdiff1d"): the set difference of two 1D arrays.

Examples

    >>> ar1 = jnp.array([1, 2, 3, 4])
    >>> ar2 = jnp.array([3, 4, 5, 6])
    >>> jnp.intersect1d(ar1, ar2)
    Array([3, 4], dtype=int32)

Computing intersection with indices:

    >>> intersection, ar1_indices, ar2_indices = jnp.intersect1d(ar1, ar2, return_indices=True)
    >>> intersection
    Array([3, 4], dtype=int32)

`ar1_indices` gives the indices of the intersected values within `ar1`:

    >>> ar1_indices
    Array([2, 3], dtype=int32)
    >>> jnp.all(intersection == ar1[ar1_indices])
    Array(True, dtype=bool)

`ar2_indices` gives the indices of the intersected values within `ar2`:

    >>> ar2_indices
    Array([0, 1], dtype=int32)
    >>> jnp.all(intersection == ar2[ar2_indices])
    Array(True, dtype=bool)

[](jax.numpy.interp.html "previous page")

previous

jax.numpy.interp

[](jax.numpy.invert.html "next page")

next

jax.numpy.invert

Contents

- [`intersect1d()`](#jax.numpy.intersect1d)

By The JAX authors

© Copyright 2024, The JAX Authors.\
