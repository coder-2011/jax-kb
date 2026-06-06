- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.acos

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.acos.rst "Download source file")
-  .pdf

# jax.lax.acos

## Contents

- [`acos()`](#jax.lax.acos)

# jax.lax.acos[\#](#jax-lax-acos "Link to this heading")

jax.lax.acos(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L3963-L3982)[\#](#jax.lax.acos "Link to this definition")  
Elementwise arc cosine: \\\mathrm{acos}(x)\\.

This function lowers directly to the `chlo.acos` operation.

Parameters:  
**x** (*ArrayLike*) – input array. Must have floating-point or complex type.

Returns:  
Array of the same shape and dtype as `x` containing the element-wise arc cosine.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.lax.cos()`](jax.lax.cos.html#jax.lax.cos "jax.lax.cos"): elementwise cosine.

- [`jax.lax.asin()`](jax.lax.asin.html#jax.lax.asin "jax.lax.asin"): elementwise arc sine.

- [`jax.lax.atan()`](jax.lax.atan.html#jax.lax.atan "jax.lax.atan"): elementwise arc tangent.

[](jax.lax.abs.html "previous page")

previous

jax.lax.abs

[](jax.lax.acosh.html "next page")

next

jax.lax.acosh

Contents

- [`acos()`](#jax.lax.acos)

By The JAX authors

© Copyright 2024, The JAX Authors.\
