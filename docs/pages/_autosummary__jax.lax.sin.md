- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.sin

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.sin.rst "Download source file")
-  .pdf

# jax.lax.sin

## Contents

- [`sin()`](#jax.lax.sin)

# jax.lax.sin[\#](#jax-lax-sin "Link to this heading")

jax.lax.sin(*x*, *\**, *accuracy=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L665-L694)[\#](#jax.lax.sin "Link to this definition")  
Elementwise sine: \\\mathrm{sin}(x)\\.

For floating-point inputs, this function lowers directly to the [stablehlo.sine](https://openxla.org/stablehlo/spec#sine) operation. For complex inputs, it lowers to a sequence of HLO operations implementing the complex sine.

Parameters:  
- **x** (*ArrayLike*) – input array. Must have floating-point or complex type.

- **accuracy** ([*Tolerance*](../jax.lax.html#jax.lax.Tolerance "jax.lax.Tolerance") *\|* [*AccuracyMode*](../jax.lax.html#jax.lax.AccuracyMode "jax.lax.AccuracyMode") *\|* *None*) – Optional lax.Tolerance or lax.AccuracyMode object that selects the implementation of the op based on the requested accuracy. If the implementation cannot satisfy the requested tolerance, the compiler will return an error. If mode is specified and there are no multiple implementations available, the default implementation will be used.

Returns:  
Array of the same shape and dtype as `x` containing the element-wise sine.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.lax.cos()`](jax.lax.cos.html#jax.lax.cos "jax.lax.cos"): elementwise cosine.

- [`jax.lax.tan()`](jax.lax.tan.html#jax.lax.tan "jax.lax.tan"): elementwise tangent.

- [`jax.lax.asin()`](jax.lax.asin.html#jax.lax.asin "jax.lax.asin"): elementwise arc sine.

[](jax.lax.sign.html "previous page")

previous

jax.lax.sign

[](jax.lax.sinh.html "next page")

next

jax.lax.sinh

Contents

- [`sin()`](#jax.lax.sin)

By The JAX authors

© Copyright 2024, The JAX Authors.\
