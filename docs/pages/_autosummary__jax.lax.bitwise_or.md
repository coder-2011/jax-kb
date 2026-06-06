- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.bitwise_or

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.bitwise_or.rst "Download source file")
-  .pdf

# jax.lax.bitwise_or

## Contents

- [`bitwise_or()`](#jax.lax.bitwise_or)

# jax.lax.bitwise_or[\#](#jax-lax-bitwise-or "Link to this heading")

jax.lax.bitwise_or(*x*, *y*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L1047-L1073)[\#](#jax.lax.bitwise_or "Link to this definition")  
Elementwise OR: \\x \vee y\\.

This function lowers directly to the [stablehlo.or](https://openxla.org/stablehlo/spec#or) operation.

Parameters:  
- **x** (*ArrayLike*) – Input arrays. Must have matching boolean or integer dtypes. If neither is a scalar, `x` and `y` must have the same number of dimensions and be broadcast compatible.

- **y** (*ArrayLike*) – Input arrays. Must have matching boolean or integer dtypes. If neither is a scalar, `x` and `y` must have the same number of dimensions and be broadcast compatible.

Returns:  
An array of the same dtype as `x` and `y` containing the bitwise OR of each pair of broadcasted entries.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.invert()`](jax.numpy.invert.html#jax.numpy.invert "jax.numpy.invert"): NumPy wrapper for this API, also accessible via the `x`` ``|`` ``y` operator on JAX arrays.

- [`jax.lax.bitwise_not()`](jax.lax.bitwise_not.html#jax.lax.bitwise_not "jax.lax.bitwise_not"): Elementwise NOT.

- [`jax.lax.bitwise_and()`](jax.lax.bitwise_and.html#jax.lax.bitwise_and "jax.lax.bitwise_and"): Elementwise AND.

- [`jax.lax.bitwise_xor()`](jax.lax.bitwise_xor.html#jax.lax.bitwise_xor "jax.lax.bitwise_xor"): Elementwise exclusive OR.

[](jax.lax.bitwise_not.html "previous page")

previous

jax.lax.bitwise_not

[](jax.lax.bitwise_xor.html "next page")

next

jax.lax.bitwise_xor

Contents

- [`bitwise_or()`](#jax.lax.bitwise_or)

By The JAX authors

© Copyright 2024, The JAX Authors.\
