- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.reduce_max

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.reduce_max.rst "Download source file")
-  .pdf

# jax.lax.reduce_max

## Contents

- [`reduce_max()`](#jax.lax.reduce_max)

# jax.lax.reduce_max[\#](#jax-lax-reduce-max "Link to this heading")

jax.lax.reduce_max(*operand*, *axes*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L3385-L3405)[\#](#jax.lax.reduce_max "Link to this definition")  
Compute the maximum of elements over one or more array axes.

Parameters:  
- **operand** (*ArrayLike*) – array over which to compute maximum.

- **axes** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – sequence of zero or more unique integers specifying the axes over which to reduce. Each entry must satisfy `0`` ``<=`` ``axis`` ``<`` ``operand.ndim`.

Returns:  
An array of the same dtype as `operand`, with shape corresponding to the dimensions of `operand.shape` with `axes` removed.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.max()`](jax.numpy.max.html#jax.numpy.max "jax.numpy.max"): more flexible NumPy-style max-reduction API, built around [`jax.lax.reduce_max()`](#jax.lax.reduce_max "jax.lax.reduce_max").

- Other low-level [`jax.lax`](../jax.lax.html#module-jax.lax "jax.lax") reduction operators: [`jax.lax.reduce_sum()`](jax.lax.reduce_sum.html#jax.lax.reduce_sum "jax.lax.reduce_sum"), [`jax.lax.reduce_prod()`](jax.lax.reduce_prod.html#jax.lax.reduce_prod "jax.lax.reduce_prod"), [`jax.lax.reduce_min()`](jax.lax.reduce_min.html#jax.lax.reduce_min "jax.lax.reduce_min"), [`jax.lax.reduce_and()`](jax.lax.reduce_and.html#jax.lax.reduce_and "jax.lax.reduce_and"), [`jax.lax.reduce_or()`](jax.lax.reduce_or.html#jax.lax.reduce_or "jax.lax.reduce_or"), [`jax.lax.reduce_xor()`](jax.lax.reduce_xor.html#jax.lax.reduce_xor "jax.lax.reduce_xor").

[](jax.lax.reduce_and.html "previous page")

previous

jax.lax.reduce_and

[](jax.lax.reduce_min.html "next page")

next

jax.lax.reduce_min

Contents

- [`reduce_max()`](#jax.lax.reduce_max)

By The JAX authors

© Copyright 2024, The JAX Authors.\
