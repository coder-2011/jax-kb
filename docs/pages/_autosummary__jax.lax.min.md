- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.min

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.min.rst "Download source file")
-  .pdf

# jax.lax.min

## Contents

- [`min()`](#jax.lax.min)

# jax.lax.min[\#](#jax-lax-min "Link to this heading")

jax.lax.min(*x*, *y*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L1326-L1351)[\#](#jax.lax.min "Link to this definition")  
Elementwise minimum: \\\mathrm{min}(x, y)\\

This function lowers directly to the [stablehlo.minimum](https://openxla.org/stablehlo/spec#minimum) operation for non-complex inputs. For complex numbers, this uses a lexicographic comparison on the (real, imaginary) pairs.

Parameters:  
- **x** (*ArrayLike*) – Input arrays. Must have matching dtypes. If neither is a scalar, `x` and `y` must have the same rank and be broadcast compatible.

- **y** (*ArrayLike*) – Input arrays. Must have matching dtypes. If neither is a scalar, `x` and `y` must have the same rank and be broadcast compatible.

Returns:  
An array of the same dtype as `x` and `y` containing the elementwise minimum.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.minimum()`](jax.numpy.minimum.html#jax.numpy.minimum "jax.numpy.minimum"): more flexibly NumPy-style minimum.

- [`jax.lax.reduce_min()`](jax.lax.reduce_min.html#jax.lax.reduce_min "jax.lax.reduce_min"): minimum along an axis of an array.

- [`jax.lax.max()`](jax.lax.max.html#jax.lax.max "jax.lax.max"): elementwise maximum.

[](jax.lax.max.html "previous page")

previous

jax.lax.max

[](jax.lax.mul.html "next page")

next

jax.lax.mul

Contents

- [`min()`](#jax.lax.min)

By The JAX authors

© Copyright 2024, The JAX Authors.\
