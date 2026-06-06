- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.acosh

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.acosh.rst "Download source file")
-  .pdf

# jax.lax.acosh

## Contents

- [`acosh()`](#jax.lax.acosh)

# jax.lax.acosh[\#](#jax-lax-acosh "Link to this heading")

jax.lax.acosh(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L4064-L4083)[\#](#jax.lax.acosh "Link to this definition")  
Elementwise inverse hyperbolic cosine: \\\mathrm{acosh}(x)\\.

This function lowers directly to the `chlo.acosh` operation.

Parameters:  
**x** (*ArrayLike*) – input array. Must have floating-point or complex type.

Returns:  
Array of the same shape and dtype as `x` containing the element-wise inverse hyperbolic cosine.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.lax.asinh()`](jax.lax.asinh.html#jax.lax.asinh "jax.lax.asinh"): elementwise inverse hyperbolic sine.

- [`jax.lax.atanh()`](jax.lax.atanh.html#jax.lax.atanh "jax.lax.atanh"): elementwise inverse hyperbolic tangent.

- [`jax.lax.cosh()`](jax.lax.cosh.html#jax.lax.cosh "jax.lax.cosh"): elementwise hyperbolic cosine.

[](jax.lax.acos.html "previous page")

previous

jax.lax.acos

[](jax.lax.add.html "next page")

next

jax.lax.add

Contents

- [`acosh()`](#jax.lax.acosh)

By The JAX authors

© Copyright 2024, The JAX Authors.\
