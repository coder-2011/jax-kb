- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.atan2

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.atan2.rst "Download source file")
-  .pdf

# jax.lax.atan2

## Contents

- [`atan2()`](#jax.lax.atan2)

# jax.lax.atan2[\#](#jax-lax-atan2 "Link to this heading")

jax.lax.atan2(*x*, *y*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L725-L749)[\#](#jax.lax.atan2 "Link to this definition")  
Elementwise two-term arc tangent: \\\mathrm{atan}({x \over y})\\.

This function lowers directly to the [stablehlo.atan2](https://openxla.org/stablehlo/spec#atan2) operation.

Parameters:  
- **x** (*ArrayLike*) – input arrays. Must have a matching floating-point or complex dtypes. If neither is a scalar, the two arrays must have the same number of dimensions and be broadcast-compatible.

- **y** (*ArrayLike*) – input arrays. Must have a matching floating-point or complex dtypes. If neither is a scalar, the two arrays must have the same number of dimensions and be broadcast-compatible.

Returns:  
Array of the same shape and dtype as `x` and `y` containing the element-wise arc tangent of \\x \over y\\, respecting the quadrant indicated by the sign of each input.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.lax.tan()`](jax.lax.tan.html#jax.lax.tan "jax.lax.tan"): elementwise tangent.

- [`jax.lax.atan()`](jax.lax.atan.html#jax.lax.atan "jax.lax.atan"): elementwise one-term arc tangent.

[](jax.lax.atan.html "previous page")

previous

jax.lax.atan

[](jax.lax.atanh.html "next page")

next

jax.lax.atanh

Contents

- [`atan2()`](#jax.lax.atan2)

By The JAX authors

© Copyright 2024, The JAX Authors.\
