- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.tan

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.tan.rst "Download source file")
-  .pdf

# jax.lax.tan

## Contents

- [`tan()`](#jax.lax.tan)

# jax.lax.tan[\#](#jax-lax-tan "Link to this heading")

jax.lax.tan(*x*, *\**, *accuracy=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L3914-L3942)[\#](#jax.lax.tan "Link to this definition")  
Elementwise tangent: \\\mathrm{tan}(x)\\.

This function lowers directly to the [stablehlo.tangent](https://openxla.org/stablehlo/spec#tangent) operation.

Parameters:  
- **x** (*ArrayLike*) – input array. Must have floating-point or complex type.

- **accuracy** ([*Tolerance*](../jax.lax.html#jax.lax.Tolerance "jax.lax.Tolerance") *\|* [*AccuracyMode*](../jax.lax.html#jax.lax.AccuracyMode "jax.lax.AccuracyMode") *\|* *None*) – Optional lax.Tolerance or lax.AccuracyMode object that selects the implementation of the op based on the requested accuracy. If the implementation cannot satisfy the requested tolerance, the compiler will return an error. If mode is specified and there are no multiple implementations available, the default implementation will be used.

Returns:  
Array of the same shape and dtype as `x` containing the element-wise tangent.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.lax.cos()`](jax.lax.cos.html#jax.lax.cos "jax.lax.cos"): elementwise cosine.

- [`jax.lax.sin()`](jax.lax.sin.html#jax.lax.sin "jax.lax.sin"): elementwise sine.

- [`jax.lax.atan()`](jax.lax.atan.html#jax.lax.atan "jax.lax.atan"): elementwise arc tangent.

- [`jax.lax.atan2()`](jax.lax.atan2.html#jax.lax.atan2 "jax.lax.atan2"): elementwise 2-term arc tangent.

[](jax.lax.sub.html "previous page")

previous

jax.lax.sub

[](jax.lax.tanh.html "next page")

next

jax.lax.tanh

Contents

- [`tan()`](#jax.lax.tan)

By The JAX authors

© Copyright 2024, The JAX Authors.\
