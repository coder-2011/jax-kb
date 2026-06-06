- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.bitwise_xor

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.bitwise_xor.rst "Download source file")
-  .pdf

# jax.lax.bitwise_xor

## Contents

- [`bitwise_xor()`](#jax.lax.bitwise_xor)

# jax.lax.bitwise_xor[\#](#jax-lax-bitwise-xor "Link to this heading")

jax.lax.bitwise_xor(*x*, *y*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L1074-L1100)[\#](#jax.lax.bitwise_xor "Link to this definition")  
Elementwise exclusive OR: \\x \oplus y\\.

This function lowers directly to the [stablehlo.xor](https://openxla.org/stablehlo/spec#xor) operation.

Parameters:  
- **x** (*ArrayLike*) – Input arrays. Must have matching boolean or integer dtypes. If neither is a scalar, `x` and `y` must have the same number of dimensions and be broadcast compatible.

- **y** (*ArrayLike*) – Input arrays. Must have matching boolean or integer dtypes. If neither is a scalar, `x` and `y` must have the same number of dimensions and be broadcast compatible.

Returns:  
An array of the same dtype as `x` and `y` containing the bitwise XOR of each pair of broadcasted entries.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.bitwise_xor()`](jax.numpy.bitwise_xor.html#jax.numpy.bitwise_xor "jax.numpy.bitwise_xor"): NumPy wrapper for this API, also accessible via the `x`` ``^`` ``y` operator on JAX arrays.

- [`jax.lax.bitwise_not()`](jax.lax.bitwise_not.html#jax.lax.bitwise_not "jax.lax.bitwise_not"): Elementwise NOT.

- [`jax.lax.bitwise_and()`](jax.lax.bitwise_and.html#jax.lax.bitwise_and "jax.lax.bitwise_and"): Elementwise AND.

- [`jax.lax.bitwise_or()`](jax.lax.bitwise_or.html#jax.lax.bitwise_or "jax.lax.bitwise_or"): Elementwise OR.

[](jax.lax.bitwise_or.html "previous page")

previous

jax.lax.bitwise_or

[](jax.lax.population_count.html "next page")

next

jax.lax.population_count

Contents

- [`bitwise_xor()`](#jax.lax.bitwise_xor)

By The JAX authors

© Copyright 2024, The JAX Authors.\
