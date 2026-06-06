- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.atan

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.atan.rst "Download source file")
-  .pdf

# jax.lax.atan

## Contents

- [`atan()`](#jax.lax.atan)

# jax.lax.atan[\#](#jax-lax-atan "Link to this heading")

jax.lax.atan(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L3983-L4003)[\#](#jax.lax.atan "Link to this definition")  
Elementwise arc tangent: \\\mathrm{atan}(x)\\.

This function lowers directly to the `chlo.atan` operation.

Parameters:  
**x** (*ArrayLike*) – input array. Must have floating-point or complex type.

Returns:  
Array of the same shape and dtype as `x` containing the element-wise arc tangent.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.lax.tan()`](jax.lax.tan.html#jax.lax.tan "jax.lax.tan"): elementwise tangent.

- [`jax.lax.acos()`](jax.lax.acos.html#jax.lax.acos "jax.lax.acos"): elementwise arc cosine.

- [`jax.lax.asin()`](jax.lax.asin.html#jax.lax.asin "jax.lax.asin"): elementwise arc sine.

- [`jax.lax.atan2()`](jax.lax.atan2.html#jax.lax.atan2 "jax.lax.atan2"): elementwise 2-term arc tangent.

[](jax.lax.asinh.html "previous page")

previous

jax.lax.asinh

[](jax.lax.atan2.html "next page")

next

jax.lax.atan2

Contents

- [`atan()`](#jax.lax.atan)

By The JAX authors

© Copyright 2024, The JAX Authors.\
