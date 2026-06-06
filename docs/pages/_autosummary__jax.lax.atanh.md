- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.atanh

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.atanh.rst "Download source file")
-  .pdf

# jax.lax.atanh

## Contents

- [`atanh()`](#jax.lax.atanh)

# jax.lax.atanh[\#](#jax-lax-atanh "Link to this heading")

jax.lax.atanh(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L4084-L4103)[\#](#jax.lax.atanh "Link to this definition")  
Elementwise inverse hyperbolic tangent: \\\mathrm{atanh}(x)\\.

This function lowers directly to the `chlo.atanh` operation.

Parameters:  
**x** (*ArrayLike*) – input array. Must have floating-point or complex type.

Returns:  
Array of the same shape and dtype as `x` containing the element-wise inverse hyperbolic tangent.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.lax.acosh()`](jax.lax.acosh.html#jax.lax.acosh "jax.lax.acosh"): elementwise inverse hyperbolic cosine.

- [`jax.lax.asinh()`](jax.lax.asinh.html#jax.lax.asinh "jax.lax.asinh"): elementwise inverse hyperbolic sine.

- [`jax.lax.tanh()`](jax.lax.tanh.html#jax.lax.tanh "jax.lax.tanh"): elementwise hyperbolic tangent.

[](jax.lax.atan2.html "previous page")

previous

jax.lax.atan2

[](jax.lax.batch_matmul.html "next page")

next

jax.lax.batch_matmul

Contents

- [`atanh()`](#jax.lax.atanh)

By The JAX authors

© Copyright 2024, The JAX Authors.\
