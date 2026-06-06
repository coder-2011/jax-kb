- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.expm1

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.expm1.rst "Download source file")
-  .pdf

# jax.lax.expm1

## Contents

- [`expm1()`](#jax.lax.expm1)

# jax.lax.expm1[\#](#jax-lax-expm1 "Link to this heading")

jax.lax.expm1(*x*, *\**, *accuracy=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L528-L556)[\#](#jax.lax.expm1 "Link to this definition")  
Elementwise \\e^{x} - 1\\.

This function lowers directly to the [stablehlo.exponential_minus_one](https://openxla.org/stablehlo/spec#exponential_minus_one) operation. Compared to the naive expression `lax.exp(x)`` ``-`` ``1`, it is more accurate for `x` near zero.

Parameters:  
- **x** (*ArrayLike*) – input array. Must have floating-point or complex type.

- **accuracy** ([*Tolerance*](../jax.lax.html#jax.lax.Tolerance "jax.lax.Tolerance") *\|* [*AccuracyMode*](../jax.lax.html#jax.lax.AccuracyMode "jax.lax.AccuracyMode") *\|* *None*) – Optional lax.Tolerance or lax.AccuracyMode object that selects the implementation of the op based on the requested accuracy. If the implementation cannot satisfy the requested tolerance, the compiler will return an error. If mode is specified and there are no multiple implementations available, the default implementation will be used.

Returns:  
Array of the same shape and dtype as `x` containing the element-wise exponential minus 1.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.lax.exp()`](jax.lax.exp.html#jax.lax.exp "jax.lax.exp"): elementwise exponentional: \\e^x\\.

- [`jax.lax.log1p()`](jax.lax.log1p.html#jax.lax.log1p "jax.lax.log1p"): elementwise \\\mathrm{log}(1 + x)\\.

[](jax.lax.expand_dims.html "previous page")

previous

jax.lax.expand_dims

[](jax.lax.fft.html "next page")

next

jax.lax.fft

Contents

- [`expm1()`](#jax.lax.expm1)

By The JAX authors

© Copyright 2024, The JAX Authors.\
