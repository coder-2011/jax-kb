- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.setxor1d

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.setxor1d.rst "Download source file")
-  .pdf

# jax.numpy.setxor1d

## Contents

- [`setxor1d()`](#jax.numpy.setxor1d)

# jax.numpy.setxor1d[\#](#jax-numpy-setxor1d "Link to this heading")

jax.numpy.setxor1d(*ar1*, *ar2*, *assume_unique=False*, *\**, *size=None*, *fill_value=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/setops.py#L287-L341)[\#](#jax.numpy.setxor1d "Link to this definition")  
Compute the set-wise xor of elements in two arrays.

JAX implementation of [`numpy.setxor1d()`](https://numpy.org/doc/stable/reference/generated/numpy.setxor1d.html#numpy.setxor1d "(in NumPy v2.4)").

Because the size of the output of `setxor1d` is data-dependent, the function is not compatible with JIT or other JAX transformations.

Parameters:  
- **ar1** (*ArrayLike*) – first array of values to intersect.

- **ar2** (*ArrayLike*) – second array of values to intersect.

- **assume_unique** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – if True, assume the input arrays contain unique values. This allows a more efficient implementation, but if `assume_unique` is True and the input arrays contain duplicates, the behavior is undefined. default: False.

- **size** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – if specified, return only the first `size` sorted elements. If there are fewer elements than `size` indicates, the return value will be padded with `fill_value`, and returned indices will be padded with an out-of-bound index.

- **fill_value** (*ArrayLike* *\|* *None*) – when `size` is specified and there are fewer than the indicated number of elements, fill the remaining entries `fill_value`. Defaults to the smallest value in the xor result.

Returns:  
An array of values that are found in exactly one of the input arrays.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.intersect1d()`](jax.numpy.intersect1d.html#jax.numpy.intersect1d "jax.numpy.intersect1d"): the set intersection of two 1D arrays.

- [`jax.numpy.union1d()`](jax.numpy.union1d.html#jax.numpy.union1d "jax.numpy.union1d"): the set union of two 1D arrays.

- [`jax.numpy.setdiff1d()`](jax.numpy.setdiff1d.html#jax.numpy.setdiff1d "jax.numpy.setdiff1d"): the set difference of two 1D arrays.

Examples

    >>> ar1 = jnp.array([1, 2, 3, 4])
    >>> ar2 = jnp.array([3, 4, 5, 6])
    >>> jnp.setxor1d(ar1, ar2)
    Array([1, 2, 5, 6], dtype=int32)

[](jax.numpy.setdiff1d.html "previous page")

previous

jax.numpy.setdiff1d

[](jax.numpy.shape.html "next page")

next

jax.numpy.shape

Contents

- [`setxor1d()`](#jax.numpy.setxor1d)

By The JAX authors

© Copyright 2024, The JAX Authors.\
