- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.reduce_xor

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.reduce_xor.rst "Download source file")
-  .pdf

# jax.lax.reduce_xor

## Contents

- [`reduce_xor()`](#jax.lax.reduce_xor)

# jax.lax.reduce_xor[\#](#jax-lax-reduce-xor "Link to this heading")

jax.lax.reduce_xor(*operand*, *axes*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L3471-L3492)[\#](#jax.lax.reduce_xor "Link to this definition")  
Compute the bitwise XOR of elements over one or more array axes.

Parameters:  
- **operand** (*ArrayLike*) – array over which to compute the reduction. Must have boolean or integer dtype.

- **axes** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – sequence of zero or more unique integers specifying the axes over which to reduce. Each entry must satisfy `0`` ``<=`` ``axis`` ``<`` ``operand.ndim`.

Returns:  
An array of the same dtype as `operand`, with shape corresponding to the dimensions of `operand.shape` with `axes` removed.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- `jax.numpy.bitwise_xor.reduce()`: more flexible NumPy-style logical reduction API, built around [`jax.lax.reduce_xor()`](#jax.lax.reduce_xor "jax.lax.reduce_xor").

- Other low-level [`jax.lax`](../jax.lax.html#module-jax.lax "jax.lax") reduction operators: [`jax.lax.reduce_sum()`](jax.lax.reduce_sum.html#jax.lax.reduce_sum "jax.lax.reduce_sum"), [`jax.lax.reduce_prod()`](jax.lax.reduce_prod.html#jax.lax.reduce_prod "jax.lax.reduce_prod"), [`jax.lax.reduce_max()`](jax.lax.reduce_max.html#jax.lax.reduce_max "jax.lax.reduce_max"), [`jax.lax.reduce_min()`](jax.lax.reduce_min.html#jax.lax.reduce_min "jax.lax.reduce_min"), [`jax.lax.reduce_and()`](jax.lax.reduce_and.html#jax.lax.reduce_and "jax.lax.reduce_and"), [`jax.lax.reduce_or()`](jax.lax.reduce_or.html#jax.lax.reduce_or "jax.lax.reduce_or").

[](jax.lax.reduce_window.html "previous page")

previous

jax.lax.reduce_window

[](jax.lax.rem.html "next page")

next

jax.lax.rem

Contents

- [`reduce_xor()`](#jax.lax.reduce_xor)

By The JAX authors

© Copyright 2024, The JAX Authors.\
