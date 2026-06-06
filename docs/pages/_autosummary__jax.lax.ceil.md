- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.ceil

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.ceil.rst "Download source file")
-  .pdf

# jax.lax.ceil

## Contents

- [`ceil()`](#jax.lax.ceil)

# jax.lax.ceil[\#](#jax-lax-ceil "Link to this heading")

jax.lax.ceil(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L360-L380)[\#](#jax.lax.ceil "Link to this definition")  
Elementwise ceiling: \\\left\lceil x \right\rceil\\.

This function lowers directly to the [stablehlo.ceil](https://openxla.org/stablehlo/spec#ceil) operation.

Parameters:  
**x** (*ArrayLike*) – input array. Must have floating-point type.

Returns:  
Array of same shape and dtype as `x`, containing values rounded to the next integer toward positive infinity.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.lax.floor()`](jax.lax.floor.html#jax.lax.floor "jax.lax.floor"): round to the next integer toward negative infinity

- [`jax.lax.round()`](jax.lax.round.html#jax.lax.round "jax.lax.round"): round to the nearest integer

[](jax.lax.cbrt.html "previous page")

previous

jax.lax.cbrt

[](jax.lax.clamp.html "next page")

next

jax.lax.clamp

Contents

- [`ceil()`](#jax.lax.ceil)

By The JAX authors

© Copyright 2024, The JAX Authors.\
