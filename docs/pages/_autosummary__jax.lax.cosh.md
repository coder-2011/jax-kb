- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.cosh

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.cosh.rst "Download source file")
-  .pdf

# jax.lax.cosh

## Contents

- [`cosh()`](#jax.lax.cosh)

# jax.lax.cosh[\#](#jax-lax-cosh "Link to this heading")

jax.lax.cosh(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L4024-L4043)[\#](#jax.lax.cosh "Link to this definition")  
Elementwise hyperbolic cosine: \\\mathrm{cosh}(x)\\.

This function lowers directly to the `chlo.cosh` operation.

Parameters:  
**x** (*ArrayLike*) – input array. Must have floating-point or complex type.

Returns:  
Array of the same shape and dtype as `x` containing the element-wise hyperbolic cosine.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.lax.acosh()`](jax.lax.acosh.html#jax.lax.acosh "jax.lax.acosh"): elementwise inverse hyperbolic cosine.

- [`jax.lax.sinh()`](jax.lax.sinh.html#jax.lax.sinh "jax.lax.sinh"): elementwise hyperbolic sine.

- [`jax.lax.tanh()`](jax.lax.tanh.html#jax.lax.tanh "jax.lax.tanh"): elementwise hyperbolic tangent.

[](jax.lax.cos.html "previous page")

previous

jax.lax.cos

[](jax.lax.cumlogsumexp.html "next page")

next

jax.lax.cumlogsumexp

Contents

- [`cosh()`](#jax.lax.cosh)

By The JAX authors

© Copyright 2024, The JAX Authors.\
