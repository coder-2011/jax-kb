- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.sub

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.sub.rst "Download source file")
-  .pdf

# jax.lax.sub

## Contents

- [`sub()`](#jax.lax.sub)

# jax.lax.sub[\#](#jax-lax-sub "Link to this heading")

jax.lax.sub(*x*, *y*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L1166-L1189)[\#](#jax.lax.sub "Link to this definition")  
Elementwise subtraction: \\x - y\\.

This function lowers directly to the [stablehlo.subtract](https://openxla.org/stablehlo/spec#subtract) operation.

Parameters:  
- **x** (*ArrayLike*) – Input arrays. Must have matching numerical dtypes. If neither is a scalar, `x` and `y` must have the same number of dimensions and be broadcast compatible.

- **y** (*ArrayLike*) – Input arrays. Must have matching numerical dtypes. If neither is a scalar, `x` and `y` must have the same number of dimensions and be broadcast compatible.

Returns:  
An array of the same dtype as `x` and `y` containing the difference of each pair of broadcasted entries.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.subtract()`](jax.numpy.subtract.html#jax.numpy.subtract "jax.numpy.subtract"): NumPy-style subtraction supporting inputs with mixed dtypes and ranks.

[](jax.lax.stage.html "previous page")

previous

jax.lax.stage

[](jax.lax.tan.html "next page")

next

jax.lax.tan

Contents

- [`sub()`](#jax.lax.sub)

By The JAX authors

© Copyright 2024, The JAX Authors.\
