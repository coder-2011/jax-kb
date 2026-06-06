- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.cbrt

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.cbrt.rst "Download source file")
-  .pdf

# jax.lax.cbrt

## Contents

- [`cbrt()`](#jax.lax.cbrt)

# jax.lax.cbrt[\#](#jax-lax-cbrt "Link to this heading")

jax.lax.cbrt(*x*, *\**, *accuracy=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L969-L995)[\#](#jax.lax.cbrt "Link to this definition")  
Elementwise cube root: \\\sqrt\[3\]{x}\\.

This function lowers directly to the [stablehlo.cbrt](https://openxla.org/stablehlo/spec#cbrt) operation.

Parameters:  
- **x** (*ArrayLike*) – Input array. Must have floating or complex dtype.

- **accuracy** ([*Tolerance*](../jax.lax.html#jax.lax.Tolerance "jax.lax.Tolerance") *\|* [*AccuracyMode*](../jax.lax.html#jax.lax.AccuracyMode "jax.lax.AccuracyMode") *\|* *None*) – Optional lax.Tolerance or lax.AccuracyMode object that selects the implementation of the op based on the requested accuracy. If the implementation cannot satisfy the requested tolerance, the compiler will return an error. If mode is specified and there are no multiple implementations available, the default implementation will be used.

Returns:  
An array of the same shape and dtype as `x` containing the cube root.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

[`jax.lax.pow()`](jax.lax.pow.html#jax.lax.pow "jax.lax.pow"): Elementwise power. [`jax.lax.sqrt()`](jax.lax.sqrt.html#jax.lax.sqrt "jax.lax.sqrt"): Elementwise square root. [`jax.lax.rsqrt()`](jax.lax.rsqrt.html#jax.lax.rsqrt "jax.lax.rsqrt"): Elementwise reciporical square root.

[](jax.lax.broadcasted_iota.html "previous page")

previous

jax.lax.broadcasted_iota

[](jax.lax.ceil.html "next page")

next

jax.lax.ceil

Contents

- [`cbrt()`](#jax.lax.cbrt)

By The JAX authors

© Copyright 2024, The JAX Authors.\
