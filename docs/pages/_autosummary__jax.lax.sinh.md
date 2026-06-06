- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.sinh

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.sinh.rst "Download source file")
-  .pdf

# jax.lax.sinh

## Contents

- [`sinh()`](#jax.lax.sinh)

# jax.lax.sinh[\#](#jax-lax-sinh "Link to this heading")

jax.lax.sinh(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L4004-L4023)[\#](#jax.lax.sinh "Link to this definition")  
Elementwise hyperbolic sine: \\\mathrm{sinh}(x)\\.

This function lowers directly to the `chlo.sinh` operation.

Parameters:  
**x** (*ArrayLike*) – input array. Must have floating-point or complex type.

Returns:  
Array of the same shape and dtype as `x` containing the element-wise hyperbolic sine.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.lax.asinh()`](jax.lax.asinh.html#jax.lax.asinh "jax.lax.asinh"): elementwise inverse hyperbolic sine.

- [`jax.lax.cosh()`](jax.lax.cosh.html#jax.lax.cosh "jax.lax.cosh"): elementwise hyperbolic cosine.

- [`jax.lax.tanh()`](jax.lax.tanh.html#jax.lax.tanh "jax.lax.tanh"): elementwise hyperbolic tangent.

[](jax.lax.sin.html "previous page")

previous

jax.lax.sin

[](jax.lax.slice.html "next page")

next

jax.lax.slice

Contents

- [`sinh()`](#jax.lax.sinh)

By The JAX authors

© Copyright 2024, The JAX Authors.\
