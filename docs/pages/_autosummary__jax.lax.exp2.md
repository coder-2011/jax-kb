- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.exp2

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.exp2.rst "Download source file")
-  .pdf

# jax.lax.exp2

## Contents

- [`exp2()`](#jax.lax.exp2)

# jax.lax.exp2[\#](#jax-lax-exp2 "Link to this heading")

jax.lax.exp2(*x*, *\**, *accuracy=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L499-L527)[\#](#jax.lax.exp2 "Link to this definition")  
Elementwise base-2 exponential: \\2^x\\.

This function is implemented in terms of the [stablehlo.exponential](https://openxla.org/stablehlo/spec#exponential) and [stablehlo.multiply](https://openxla.org/stablehlo/spec#multiply) operations.

Parameters:  
- **x** (*ArrayLike*) – input array. Must have floating-point or complex type.

- **accuracy** ([*Tolerance*](../jax.lax.html#jax.lax.Tolerance "jax.lax.Tolerance") *\|* [*AccuracyMode*](../jax.lax.html#jax.lax.AccuracyMode "jax.lax.AccuracyMode") *\|* *None*) – Optional lax.Tolerance or lax.AccuracyMode object that selects the implementation of the op based on the requested accuracy. If the implementation cannot satisfy the requested tolerance, the compiler will return an error. If mode is specified and there are no multiple implementations available, the default implementation will be used.

Returns:  
Array of the same shape and dtype as `x` containing the element-wise base-2 exponential.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.lax.exp()`](jax.lax.exp.html#jax.lax.exp "jax.lax.exp"): elementwise exponentional: \\e^x\\.

- [`jax.lax.log()`](jax.lax.log.html#jax.lax.log "jax.lax.log"): elementwise natural logarithm: \\\mathrm{log}(x)\\.

[](jax.lax.exp.html "previous page")

previous

jax.lax.exp

[](jax.lax.expand_dims.html "next page")

next

jax.lax.expand_dims

Contents

- [`exp2()`](#jax.lax.exp2)

By The JAX authors

© Copyright 2024, The JAX Authors.\
