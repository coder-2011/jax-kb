- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.max

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.max.rst "Download source file")
-  .pdf

# jax.lax.max

## Contents

- [`max()`](#jax.lax.max)

# jax.lax.max[\#](#jax-lax-max "Link to this heading")

jax.lax.max(*x*, *y*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L1300-L1325)[\#](#jax.lax.max "Link to this definition")  
Elementwise maximum: \\\mathrm{max}(x, y)\\.

This function lowers directly to the [stablehlo.maximum](https://openxla.org/stablehlo/spec#maximum) operation for non-complex inputs. For complex numbers, this uses a lexicographic comparison on the (real, imaginary) pairs.

Parameters:  
- **x** (*ArrayLike*) – Input arrays. Must have matching dtypes. If neither is a scalar, `x` and `y` must have the same rank and be broadcast compatible.

- **y** (*ArrayLike*) – Input arrays. Must have matching dtypes. If neither is a scalar, `x` and `y` must have the same rank and be broadcast compatible.

Returns:  
An array of the same dtype as `x` and `y` containing the elementwise maximum.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.maximum()`](jax.numpy.maximum.html#jax.numpy.maximum "jax.numpy.maximum"): more flexibly NumPy-style maximum.

- [`jax.lax.reduce_max()`](jax.lax.reduce_max.html#jax.lax.reduce_max "jax.lax.reduce_max"): maximum along an axis of an array.

- [`jax.lax.min()`](jax.lax.min.html#jax.lax.min "jax.lax.min"): elementwise minimum.

[](jax.lax.lt.html "previous page")

previous

jax.lax.lt

[](jax.lax.min.html "next page")

next

jax.lax.min

Contents

- [`max()`](#jax.lax.max)

By The JAX authors

© Copyright 2024, The JAX Authors.\
