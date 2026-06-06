- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.bitwise_and

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.bitwise_and.rst "Download source file")
-  .pdf

# jax.lax.bitwise_and

## Contents

- [`bitwise_and()`](#jax.lax.bitwise_and)

# jax.lax.bitwise_and[\#](#jax-lax-bitwise-and "Link to this heading")

jax.lax.bitwise_and(*x*, *y*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L1020-L1046)[\#](#jax.lax.bitwise_and "Link to this definition")  
Elementwise AND: \\x \wedge y\\.

This function lowers directly to the [stablehlo.and](https://openxla.org/stablehlo/spec#and) operation.

Parameters:  
- **x** (*ArrayLike*) – Input arrays. Must have matching boolean or integer dtypes. If neither is a scalar, `x` and `y` must have the same number of dimensions and be broadcast compatible.

- **y** (*ArrayLike*) – Input arrays. Must have matching boolean or integer dtypes. If neither is a scalar, `x` and `y` must have the same number of dimensions and be broadcast compatible.

Returns:  
An array of the same dtype as `x` and `y` containing the bitwise AND of each pair of broadcasted entries.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.bitwise_and()`](jax.numpy.bitwise_and.html#jax.numpy.bitwise_and "jax.numpy.bitwise_and"): NumPy wrapper for this API, also accessible via the `x`` ``&`` ``y` operator on JAX arrays.

- [`jax.lax.bitwise_not()`](jax.lax.bitwise_not.html#jax.lax.bitwise_not "jax.lax.bitwise_not"): Elementwise NOT.

- [`jax.lax.bitwise_or()`](jax.lax.bitwise_or.html#jax.lax.bitwise_or "jax.lax.bitwise_or"): Elementwise OR.

- [`jax.lax.bitwise_xor()`](jax.lax.bitwise_xor.html#jax.lax.bitwise_xor "jax.lax.bitwise_xor"): Elementwise exclusive OR.

[](jax.lax.bitcast_convert_type.html "previous page")

previous

jax.lax.bitcast_convert_type

[](jax.lax.bitwise_not.html "next page")

next

jax.lax.bitwise_not

Contents

- [`bitwise_and()`](#jax.lax.bitwise_and)

By The JAX authors

© Copyright 2024, The JAX Authors.\
