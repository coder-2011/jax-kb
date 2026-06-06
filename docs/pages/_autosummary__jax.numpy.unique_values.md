- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.unique_values

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.unique_values.rst "Download source file")
-  .pdf

# jax.numpy.unique_values

## Contents

- [`unique_values()`](#jax.numpy.unique_values)

# jax.numpy.unique_values[\#](#jax-numpy-unique-values "Link to this heading")

jax.numpy.unique_values(*x*, */*, *\**, *size=None*, *fill_value=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/setops.py#L1094-L1134)[\#](#jax.numpy.unique_values "Link to this definition")  
Return unique values from x, along with indices, inverse indices, and counts.

JAX implementation of [`numpy.unique_values()`](https://numpy.org/doc/stable/reference/generated/numpy.unique_values.html#numpy.unique_values "(in NumPy v2.4)"); this is equivalent to calling [`jax.numpy.unique()`](jax.numpy.unique.html#jax.numpy.unique "jax.numpy.unique") with equal_nan set to True.

Because the size of the output of `unique_values` is data-dependent, the function is not typically compatible with [`jit()`](jax.jit.html#jax.jit "jax.jit") and other JAX transformations. The JAX version adds the optional `size` argument which must be specified statically for `jnp.unique` to be used in such contexts.

Parameters:  
- **x** (*ArrayLike*) – N-dimensional array from which unique values will be extracted.

- **size** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – if specified, return only the first `size` sorted unique elements. If there are fewer unique elements than `size` indicates, the return value will be padded with `fill_value`.

- **fill_value** (*ArrayLike* *\|* *None*) – when `size` is specified and there are fewer than the indicated number of elements, fill the remaining entries `fill_value`. Defaults to the minimum unique value.

Returns:  
An array `values` of shape `(n_unique,)` containing the unique values from `x`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.unique()`](jax.numpy.unique.html#jax.numpy.unique "jax.numpy.unique"): general function for computing unique values.

- [`jax.numpy.unique_values()`](#jax.numpy.unique_values "jax.numpy.unique_values"): compute only `values`.

- [`jax.numpy.unique_counts()`](jax.numpy.unique_counts.html#jax.numpy.unique_counts "jax.numpy.unique_counts"): compute only `values` and `counts`.

- [`jax.numpy.unique_inverse()`](jax.numpy.unique_inverse.html#jax.numpy.unique_inverse "jax.numpy.unique_inverse"): compute only `values` and `inverse`.

Examples

Here we compute the unique values in a 1D array:

    >>> x = jnp.array([3, 4, 1, 3, 1])
    >>> jnp.unique_values(x)
    Array([1, 3, 4], dtype=int32)

For examples of the `size` and `fill_value` arguments, see [`jax.numpy.unique()`](jax.numpy.unique.html#jax.numpy.unique "jax.numpy.unique").

[](jax.numpy.unique_inverse.html "previous page")

previous

jax.numpy.unique_inverse

[](jax.numpy.unpackbits.html "next page")

next

jax.numpy.unpackbits

Contents

- [`unique_values()`](#jax.numpy.unique_values)

By The JAX authors

© Copyright 2024, The JAX Authors.\
