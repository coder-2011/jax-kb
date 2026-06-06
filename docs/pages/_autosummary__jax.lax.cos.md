- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.cos

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.cos.rst "Download source file")
-  .pdf

# jax.lax.cos

## Contents

- [`cos()`](#jax.lax.cos)

# jax.lax.cos[\#](#jax-lax-cos "Link to this heading")

jax.lax.cos(*x*, *\**, *accuracy=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L695-L724)[\#](#jax.lax.cos "Link to this definition")  
Elementwise cosine: \\\mathrm{cos}(x)\\.

For floating-point inputs, this function lowers directly to the [stablehlo.cosine](https://openxla.org/stablehlo/spec#cosine) operation. For complex inputs, it lowers to a sequence of HLO operations implementing the complex cosine.

Parameters:  
- **x** (*ArrayLike*) – input array. Must have floating-point or complex type.

- **accuracy** ([*Tolerance*](../jax.lax.html#jax.lax.Tolerance "jax.lax.Tolerance") *\|* [*AccuracyMode*](../jax.lax.html#jax.lax.AccuracyMode "jax.lax.AccuracyMode") *\|* *None*) – Optional lax.Tolerance or lax.AccuracyMode object that selects the implementation of the op based on the requested accuracy. If the implementation cannot satisfy the requested tolerance, the compiler will return an error. If mode is specified and there are no multiple implementations available, the default implementation will be used.

Returns:  
Array of the same shape and dtype as `x` containing the element-wise cosine.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.lax.sin()`](jax.lax.sin.html#jax.lax.sin "jax.lax.sin"): elementwise sine.

- [`jax.lax.tan()`](jax.lax.tan.html#jax.lax.tan "jax.lax.tan"): elementwise tangent.

- [`jax.lax.acos()`](jax.lax.acos.html#jax.lax.acos "jax.lax.acos"): elementwise arc cosine.

[](jax.lax.conv_with_general_padding.html "previous page")

previous

jax.lax.conv_with_general_padding

[](jax.lax.cosh.html "next page")

next

jax.lax.cosh

Contents

- [`cos()`](#jax.lax.cos)

By The JAX authors

© Copyright 2024, The JAX Authors.\
