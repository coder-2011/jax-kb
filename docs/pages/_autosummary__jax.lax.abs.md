- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.abs

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.abs.rst "Download source file")
-  .pdf

# jax.lax.abs

## Contents

- [`abs()`](#jax.lax.abs)

# jax.lax.abs[\#](#jax-lax-abs "Link to this heading")

jax.lax.abs(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L846-L865)[\#](#jax.lax.abs "Link to this definition")  
Elementwise absolute value: \\\|x\|\\.

This function lowers directly to the [stablehlo.abs](https://openxla.org/stablehlo/spec#abs) operation.

Parameters:  
**x** (*ArrayLike*) – Input array. Must have signed integer, floating, or complex dtype.

Returns:  
An array of the same shape as `x` containing the elementwise absolute value. For complex valued input, \\a + ib\\, `abs(x)` returns \\\sqrt{a^2+b^2}\\.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.abs()`](jax.numpy.abs.html#jax.numpy.abs "jax.numpy.abs"): a more flexible NumPy-style `abs` implementation.

[](../jax.lax.html "previous page")

previous

`jax.lax` module

[](jax.lax.acos.html "next page")

next

jax.lax.acos

Contents

- [`abs()`](#jax.lax.abs)

By The JAX authors

© Copyright 2024, The JAX Authors.\
