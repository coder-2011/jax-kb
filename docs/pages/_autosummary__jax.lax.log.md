- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.log

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.log.rst "Download source file")
-  .pdf

# jax.lax.log

## Contents

- [`log()`](#jax.lax.log)

# jax.lax.log[\#](#jax-lax-log "Link to this heading")

jax.lax.log(*x*, *\**, *accuracy=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L557-L582)[\#](#jax.lax.log "Link to this definition")  
Elementwise natural logarithm: \\\mathrm{log}(x)\\.

This function lowers directly to the [stablehlo.log](https://openxla.org/stablehlo/spec#log) operation.

Parameters:  
- **x** (*ArrayLike*) – input array. Must have floating-point or complex type.

- **accuracy** ([*Tolerance*](../jax.lax.html#jax.lax.Tolerance "jax.lax.Tolerance") *\|* [*AccuracyMode*](../jax.lax.html#jax.lax.AccuracyMode "jax.lax.AccuracyMode") *\|* *None*) – Optional lax.Tolerance or lax.AccuracyMode object that selects the implementation of the op based on the requested accuracy. If the implementation cannot satisfy the requested tolerance, the compiler will return an error. If mode is specified and there are no multiple implementations available, the default implementation will be used.

Returns:  
Array of the same shape and dtype as `x` containing the element-wise natural logarithm.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.lax.exp()`](jax.lax.exp.html#jax.lax.exp "jax.lax.exp"): elementwise exponentional: \\e^x\\.

[](jax.lax.lgamma.html "previous page")

previous

jax.lax.lgamma

[](jax.lax.log1p.html "next page")

next

jax.lax.log1p

Contents

- [`log()`](#jax.lax.log)

By The JAX authors

© Copyright 2024, The JAX Authors.\
