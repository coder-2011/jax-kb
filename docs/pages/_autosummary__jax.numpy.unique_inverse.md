- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.unique_inverse

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.unique_inverse.rst "Download source file")
-  .pdf

# jax.numpy.unique_inverse

## Contents

- [`unique_inverse()`](#jax.numpy.unique_inverse)

# jax.numpy.unique_inverse[\#](#jax-numpy-unique-inverse "Link to this heading")

jax.numpy.unique_inverse(*x*, */*, *\**, *size=None*, *fill_value=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/setops.py#L1028-L1092)[\#](#jax.numpy.unique_inverse "Link to this definition")  
Return unique values from x, along with indices, inverse indices, and counts.

JAX implementation of [`numpy.unique_inverse()`](https://numpy.org/doc/stable/reference/generated/numpy.unique_inverse.html#numpy.unique_inverse "(in NumPy v2.4)"); this is equivalent to calling [`jax.numpy.unique()`](jax.numpy.unique.html#jax.numpy.unique "jax.numpy.unique") with return_inverse and equal_nan set to True.

Because the size of the output of `unique_inverse` is data-dependent, the function is not typically compatible with [`jit()`](jax.jit.html#jax.jit "jax.jit") and other JAX transformations. The JAX version adds the optional `size` argument which must be specified statically for `jnp.unique` to be used in such contexts.

Parameters:  
- **x** (*ArrayLike*) – N-dimensional array from which unique values will be extracted.

- **size** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – if specified, return only the first `size` sorted unique elements. If there are fewer unique elements than `size` indicates, the return value will be padded with `fill_value`.

- **fill_value** (*ArrayLike* *\|* *None*) – when `size` is specified and there are fewer than the indicated number of elements, fill the remaining entries `fill_value`. Defaults to the minimum unique value.

Returns:  
- `values`:  
  an array of shape `(n_unique,)` containing the unique values from `x`.

- `inverse_indices`:  
  An array of shape `x.shape`. Contains the indices within `values` of each value in `x`. For 1D inputs, `values[inverse_indices]` is equivalent to `x`.

Return type:  
A tuple `(values,`` ``indices,`` ``inverse_indices,`` ``counts)`, with the following properties

See also

- [`jax.numpy.unique()`](jax.numpy.unique.html#jax.numpy.unique "jax.numpy.unique"): general function for computing unique values.

- [`jax.numpy.unique_values()`](jax.numpy.unique_values.html#jax.numpy.unique_values "jax.numpy.unique_values"): compute only `values`.

- [`jax.numpy.unique_counts()`](jax.numpy.unique_counts.html#jax.numpy.unique_counts "jax.numpy.unique_counts"): compute only `values` and `counts`.

- [`jax.numpy.unique_all()`](jax.numpy.unique_all.html#jax.numpy.unique_all "jax.numpy.unique_all"): compute `values`, `indices`, `inverse_indices`, and `counts`.

Examples

Here we compute the unique values in a 1D array:

    >>> x = jnp.array([3, 4, 1, 3, 1])
    >>> result = jnp.unique_inverse(x)

The result is a [`NamedTuple`](https://docs.python.org/3/library/typing.html#typing.NamedTuple "(in Python v3.14)") with two named attributes. The `values` attribute contains the unique values from the array:

    >>> result.values
    Array([1, 3, 4], dtype=int32)

The `indices` attribute contains the indices of the unique `values` within the input array:

The `inverse_indices` attribute contains the indices of the input within `values`:

    >>> result.inverse_indices
    Array([1, 2, 0, 1, 0], dtype=int32)
    >>> jnp.all(x == result.values[result.inverse_indices])
    Array(True, dtype=bool)

For examples of the `size` and `fill_value` arguments, see [`jax.numpy.unique()`](jax.numpy.unique.html#jax.numpy.unique "jax.numpy.unique").

[](jax.numpy.unique_counts.html "previous page")

previous

jax.numpy.unique_counts

[](jax.numpy.unique_values.html "next page")

next

jax.numpy.unique_values

Contents

- [`unique_inverse()`](#jax.numpy.unique_inverse)

By The JAX authors

© Copyright 2024, The JAX Authors.\
