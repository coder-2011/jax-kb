- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.reduce_sum

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.reduce_sum.rst "Download source file")
-  .pdf

# jax.lax.reduce_sum

## Contents

- [`reduce_sum()`](#jax.lax.reduce_sum)

# jax.lax.reduce_sum[\#](#jax-lax-reduce-sum "Link to this heading")

jax.lax.reduce_sum(*operand*, *axes*, *\**, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L3331-L3358)[\#](#jax.lax.reduce_sum "Link to this definition")  
Compute the sum of elements over one or more array axes.

Parameters:  
- **operand** (*ArrayLike*) – array over which to sum. Must have numerical dtype.

- **axes** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – sequence of zero or more unique integers specifying the axes over which to sum. Each entry must satisfy `0`` ``<=`` ``axis`` ``<`` ``operand.ndim`.

Returns:  
An array of the same dtype as `operand`, with shape corresponding to the dimensions of `operand.shape` with `axes` removed.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Notes

Unlike [`jax.numpy.sum()`](jax.numpy.sum.html#jax.numpy.sum "jax.numpy.sum"), [`jax.lax.reduce_sum()`](#jax.lax.reduce_sum "jax.lax.reduce_sum") does not upcast narrow-width types for accumulation, so sums of 8-bit or 16-bit types may be subject to rounding errors.

See also

- [`jax.numpy.sum()`](jax.numpy.sum.html#jax.numpy.sum "jax.numpy.sum"): more flexible NumPy-style summation API, built around [`jax.lax.reduce_sum()`](#jax.lax.reduce_sum "jax.lax.reduce_sum").

- Other low-level [`jax.lax`](../jax.lax.html#module-jax.lax "jax.lax") reduction operators: [`jax.lax.reduce_prod()`](jax.lax.reduce_prod.html#jax.lax.reduce_prod "jax.lax.reduce_prod"), [`jax.lax.reduce_max()`](jax.lax.reduce_max.html#jax.lax.reduce_max "jax.lax.reduce_max"), [`jax.lax.reduce_min()`](jax.lax.reduce_min.html#jax.lax.reduce_min "jax.lax.reduce_min"), [`jax.lax.reduce_and()`](jax.lax.reduce_and.html#jax.lax.reduce_and "jax.lax.reduce_and"), [`jax.lax.reduce_or()`](jax.lax.reduce_or.html#jax.lax.reduce_or "jax.lax.reduce_or"), [`jax.lax.reduce_xor()`](jax.lax.reduce_xor.html#jax.lax.reduce_xor "jax.lax.reduce_xor").

[](jax.lax.reduce_prod.html "previous page")

previous

jax.lax.reduce_prod

[](jax.lax.reduce_window.html "next page")

next

jax.lax.reduce_window

Contents

- [`reduce_sum()`](#jax.lax.reduce_sum)

By The JAX authors

© Copyright 2024, The JAX Authors.\
