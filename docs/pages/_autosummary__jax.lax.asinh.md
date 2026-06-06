- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.asinh

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.asinh.rst "Download source file")
-  .pdf

# jax.lax.asinh

## Contents

- [`asinh()`](#jax.lax.asinh)

# jax.lax.asinh[\#](#jax-lax-asinh "Link to this heading")

jax.lax.asinh(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L4044-L4063)[\#](#jax.lax.asinh "Link to this definition")  
Elementwise inverse hyperbolic sine: \\\mathrm{asinh}(x)\\.

This function lowers directly to the `chlo.asinh` operation.

Parameters:  
**x** (*ArrayLike*) – input array. Must have floating-point or complex type.

Returns:  
Array of the same shape and dtype as `x` containing the element-wise inverse hyperbolic sine.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.lax.acosh()`](jax.lax.acosh.html#jax.lax.acosh "jax.lax.acosh"): elementwise inverse hyperbolic cosine.

- [`jax.lax.atanh()`](jax.lax.atanh.html#jax.lax.atanh "jax.lax.atanh"): elementwise inverse hyperbolic tangent.

- [`jax.lax.sinh()`](jax.lax.sinh.html#jax.lax.sinh "jax.lax.sinh"): elementwise hyperbolic sine.

[](jax.lax.asin.html "previous page")

previous

jax.lax.asin

[](jax.lax.atan.html "next page")

next

jax.lax.atan

Contents

- [`asinh()`](#jax.lax.asinh)

By The JAX authors

© Copyright 2024, The JAX Authors.\
