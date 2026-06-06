- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.log1p

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.log1p.rst "Download source file")
-  .pdf

# jax.lax.log1p

## Contents

- [`log1p()`](#jax.lax.log1p)

# jax.lax.log1p[\#](#jax-lax-log1p "Link to this heading")

jax.lax.log1p(*x*, *\**, *accuracy=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L583-L611)[\#](#jax.lax.log1p "Link to this definition")  
Elementwise \\\mathrm{log}(1 + x)\\.

This function lowers directly to the [stablehlo.log_plus_one](https://openxla.org/stablehlo/spec#log_plus_one) operation. Compared to the naive expression `lax.log(1`` ``+`` ``x)`, it is more accurate for `x` near zero.

Parameters:  
- **x** (*ArrayLike*) – input array. Must have floating-point or complex type.

- **accuracy** ([*Tolerance*](../jax.lax.html#jax.lax.Tolerance "jax.lax.Tolerance") *\|* [*AccuracyMode*](../jax.lax.html#jax.lax.AccuracyMode "jax.lax.AccuracyMode") *\|* *None*) – Optional lax.Tolerance or lax.AccuracyMode object that selects the implementation of the op based on the requested accuracy. If the implementation cannot satisfy the requested tolerance, the compiler will return an error. If mode is specified and there are no multiple implementations available, the default implementation will be used.

Returns:  
Array of the same shape and dtype as `x` containing the element-wise natural logarithm of `x`` ``+`` ``1`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.lax.expm1()`](jax.lax.expm1.html#jax.lax.expm1 "jax.lax.expm1"): elementwise \\e^x - 1\\.

- [`jax.lax.log()`](jax.lax.log.html#jax.lax.log "jax.lax.log"): elementwise natural logarithm \\\mathrm{log}(x)\\.

[](jax.lax.log.html "previous page")

previous

jax.lax.log

[](jax.lax.logistic.html "next page")

next

jax.lax.logistic

Contents

- [`log1p()`](#jax.lax.log1p)

By The JAX authors

© Copyright 2024, The JAX Authors.\
