- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.exp

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.exp.rst "Download source file")
-  .pdf

# jax.lax.exp

## Contents

- [`exp()`](#jax.lax.exp)

# jax.lax.exp[\#](#jax-lax-exp "Link to this heading")

jax.lax.exp(*x*, *\**, *accuracy=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L472-L498)[\#](#jax.lax.exp "Link to this definition")  
Elementwise exponential: \\e^x\\.

This function lowers directly to the [stablehlo.exponential](https://openxla.org/stablehlo/spec#exponential) operation.

Parameters:  
- **x** (*ArrayLike*) – input array. Must have floating-point or complex type.

- **accuracy** ([*Tolerance*](../jax.lax.html#jax.lax.Tolerance "jax.lax.Tolerance") *\|* [*AccuracyMode*](../jax.lax.html#jax.lax.AccuracyMode "jax.lax.AccuracyMode") *\|* *None*) – Optional lax.Tolerance or lax.AccuracyMode object that selects the implementation of the op based on the requested accuracy. If the implementation cannot satisfy the requested tolerance, the compiler will return an error. If mode is specified and there are no multiple implementations available, the default implementation will be used.

Returns:  
Array of the same shape and dtype as `x` containing the element-wise exponential.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.lax.exp2()`](jax.lax.exp2.html#jax.lax.exp2 "jax.lax.exp2"): elementwise base-2 exponentional: \\2^x\\.

- [`jax.lax.log()`](jax.lax.log.html#jax.lax.log "jax.lax.log"): elementwise natural logarithm: \\\mathrm{log}(x)\\.

[](jax.lax.erf_inv.html "previous page")

previous

jax.lax.erf_inv

[](jax.lax.exp2.html "next page")

next

jax.lax.exp2

Contents

- [`exp()`](#jax.lax.exp)

By The JAX authors

© Copyright 2024, The JAX Authors.\
