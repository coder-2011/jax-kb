- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.floor

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.floor.rst "Download source file")
-  .pdf

# jax.lax.floor

## Contents

- [`floor()`](#jax.lax.floor)

# jax.lax.floor[\#](#jax-lax-floor "Link to this heading")

jax.lax.floor(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L339-L359)[\#](#jax.lax.floor "Link to this definition")  
Elementwise floor: \\\left\lfloor x \right\rfloor\\.

This function lowers directly to the [stablehlo.floor](https://openxla.org/stablehlo/spec#floor) operation.

Parameters:  
**x** (*ArrayLike*) – input array. Must have floating-point type.

Returns:  
Array of same shape and dtype as `x`, containing values rounded to the next integer toward negative infinity.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.lax.ceil()`](jax.lax.ceil.html#jax.lax.ceil "jax.lax.ceil"): round to the next integer toward positive infinity

- [`jax.lax.round()`](jax.lax.round.html#jax.lax.round "jax.lax.round"): round to the nearest integer

[](jax.lax.fft.html "previous page")

previous

jax.lax.fft

[](jax.lax.full.html "next page")

next

jax.lax.full

Contents

- [`floor()`](#jax.lax.floor)

By The JAX authors

© Copyright 2024, The JAX Authors.\
