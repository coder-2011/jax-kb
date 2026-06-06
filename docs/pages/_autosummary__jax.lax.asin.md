- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.asin

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.asin.rst "Download source file")
-  .pdf

# jax.lax.asin

## Contents

- [`asin()`](#jax.lax.asin)

# jax.lax.asin[\#](#jax-lax-asin "Link to this heading")

jax.lax.asin(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L3943-L3962)[\#](#jax.lax.asin "Link to this definition")  
Elementwise arc sine: \\\mathrm{asin}(x)\\.

This function lowers directly to the `chlo.asin` operation.

Parameters:  
**x** (*ArrayLike*) – input array. Must have floating-point or complex type.

Returns:  
Array of the same shape and dtype as `x` containing the element-wise arc sine.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.lax.sin()`](jax.lax.sin.html#jax.lax.sin "jax.lax.sin"): elementwise sine.

- [`jax.lax.acos()`](jax.lax.acos.html#jax.lax.acos "jax.lax.acos"): elementwise arc cosine.

- [`jax.lax.atan()`](jax.lax.atan.html#jax.lax.atan "jax.lax.atan"): elementwise arc tangent.

[](jax.lax.argmin.html "previous page")

previous

jax.lax.argmin

[](jax.lax.asinh.html "next page")

next

jax.lax.asinh

Contents

- [`asin()`](#jax.lax.asin)

By The JAX authors

© Copyright 2024, The JAX Authors.\
