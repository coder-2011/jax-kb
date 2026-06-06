- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.sqrt

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.sqrt.rst "Download source file")
-  .pdf

# jax.lax.sqrt

## Contents

- [`sqrt()`](#jax.lax.sqrt)

# jax.lax.sqrt[\#](#jax-lax-sqrt "Link to this heading")

jax.lax.sqrt(*x*, *\**, *accuracy=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L914-L940)[\#](#jax.lax.sqrt "Link to this definition")  
Elementwise square root: \\\sqrt{x}\\.

This function lowers directly to the [stablehlo.sqrt](https://openxla.org/stablehlo/spec#sqrt) operation.

Parameters:  
- **x** (*ArrayLike*) – Input array. Must have floating or complex dtype.

- **accuracy** ([*Tolerance*](../jax.lax.html#jax.lax.Tolerance "jax.lax.Tolerance") *\|* [*AccuracyMode*](../jax.lax.html#jax.lax.AccuracyMode "jax.lax.AccuracyMode") *\|* *None*) – Optional lax.Tolerance or lax.AccuracyMode object that selects the implementation of the op based on the requested accuracy. If the implementation cannot satisfy the requested tolerance, the compiler will return an error. If mode is specified and there are no multiple implementations available, the default implementation will be used.

Returns:  
An array of the same shape and dtype as `x` containing the square root.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

[`jax.lax.pow()`](jax.lax.pow.html#jax.lax.pow "jax.lax.pow"): Elementwise power. [`jax.lax.cbrt()`](jax.lax.cbrt.html#jax.lax.cbrt "jax.lax.cbrt"): Elementwise cube root. [`jax.lax.rsqrt()`](jax.lax.rsqrt.html#jax.lax.rsqrt "jax.lax.rsqrt"): Elementwise reciporical square root.

[](jax.lax.split.html "previous page")

previous

jax.lax.split

[](jax.lax.square.html "next page")

next

jax.lax.square

Contents

- [`sqrt()`](#jax.lax.sqrt)

By The JAX authors

© Copyright 2024, The JAX Authors.\
