- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.bitwise_not

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.bitwise_not.rst "Download source file")
-  .pdf

# jax.lax.bitwise_not

## Contents

- [`bitwise_not()`](#jax.lax.bitwise_not)

# jax.lax.bitwise_not[\#](#jax-lax-bitwise-not "Link to this heading")

jax.lax.bitwise_not(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L996-L1019)[\#](#jax.lax.bitwise_not "Link to this definition")  
Elementwise NOT: \\\neg x\\.

This function lowers directly to the [stablehlo.not](https://openxla.org/stablehlo/spec#not) operation.

Parameters:  
**x** (*ArrayLike*) – Input array. Must have boolean or integer dtype.

Returns:  
An array of the same shape and dtype as `x` containing the bitwise inversion of each entry.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.invert()`](jax.numpy.invert.html#jax.numpy.invert "jax.numpy.invert"): NumPy wrapper for this API, also accessible via the `~x` operator on JAX arrays.

- [`jax.lax.bitwise_and()`](jax.lax.bitwise_and.html#jax.lax.bitwise_and "jax.lax.bitwise_and"): Elementwise AND.

- [`jax.lax.bitwise_or()`](jax.lax.bitwise_or.html#jax.lax.bitwise_or "jax.lax.bitwise_or"): Elementwise OR.

- [`jax.lax.bitwise_xor()`](jax.lax.bitwise_xor.html#jax.lax.bitwise_xor "jax.lax.bitwise_xor"): Elementwise exclusive OR.

[](jax.lax.bitwise_and.html "previous page")

previous

jax.lax.bitwise_and

[](jax.lax.bitwise_or.html "next page")

next

jax.lax.bitwise_or

Contents

- [`bitwise_not()`](#jax.lax.bitwise_not)

By The JAX authors

© Copyright 2024, The JAX Authors.\
