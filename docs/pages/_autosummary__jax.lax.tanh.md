- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.tanh

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.tanh.rst "Download source file")
-  .pdf

# jax.lax.tanh

## Contents

- [`tanh()`](#jax.lax.tanh)

# jax.lax.tanh[\#](#jax-lax-tanh "Link to this heading")

jax.lax.tanh(*x*, *\**, *accuracy=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L612-L639)[\#](#jax.lax.tanh "Link to this definition")  
Elementwise hyperbolic tangent: \\\mathrm{tanh}(x)\\.

This function lowers directly to the [stablehlo.tanh](https://openxla.org/stablehlo/spec#tanh) operation.

Parameters:  
- **x** (*ArrayLike*) – input array. Must have floating-point or complex type.

- **accuracy** ([*Tolerance*](../jax.lax.html#jax.lax.Tolerance "jax.lax.Tolerance") *\|* [*AccuracyMode*](../jax.lax.html#jax.lax.AccuracyMode "jax.lax.AccuracyMode") *\|* *None*) – Optional lax.Tolerance or lax.AccuracyMode object that selects the implementation of the op based on the requested accuracy. If the implementation cannot satisfy the requested tolerance, the compiler will return an error. If mode is specified and there are no multiple implementations available, the default implementation will be used.

Returns:  
Array of the same shape and dtype as `x` containing the element-wise hyperbolic tangent.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.lax.atanh()`](jax.lax.atanh.html#jax.lax.atanh "jax.lax.atanh"): elementwise inverse hyperbolic tangent.

- [`jax.lax.cosh()`](jax.lax.cosh.html#jax.lax.cosh "jax.lax.cosh"): elementwise hyperbolic cosine.

- [`jax.lax.sinh()`](jax.lax.sinh.html#jax.lax.sinh "jax.lax.sinh"): elementwise hyperbolic sine.

[](jax.lax.tan.html "previous page")

previous

jax.lax.tan

[](jax.lax.tile.html "next page")

next

jax.lax.tile

Contents

- [`tanh()`](#jax.lax.tanh)

By The JAX authors

© Copyright 2024, The JAX Authors.\
