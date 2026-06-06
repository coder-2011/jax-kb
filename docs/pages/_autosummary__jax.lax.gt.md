- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.gt

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.gt.rst "Download source file")
-  .pdf

# jax.lax.gt

## Contents

- [`gt()`](#jax.lax.gt)

# jax.lax.gt[\#](#jax-lax-gt "Link to this heading")

jax.lax.gt(*x*, *y*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L1525-L1555)[\#](#jax.lax.gt "Link to this definition")  
Elementwise greater-than: \\x \> y\\.

This function lowers directly to the [stablehlo.compare](https://openxla.org/stablehlo/spec#compare) operation with `comparison_direction=GT` and `compare_type` set according to the input dtype.

Parameters:  
- **x** (*ArrayLike*) – Input arrays. Must have matching non-complex dtypes. If neither is a scalar, `x` and `y` must have the same number of dimensions and be broadcast compatible.

- **y** (*ArrayLike*) – Input arrays. Must have matching non-complex dtypes. If neither is a scalar, `x` and `y` must have the same number of dimensions and be broadcast compatible.

Returns:  
A boolean array of shape `lax.broadcast_shapes(x.shape,`` ``y.shape)` containing the elementwise greater-than comparison.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.greater()`](jax.numpy.greater.html#jax.numpy.greater "jax.numpy.greater"): NumPy wrapper for this API, also accessible via the `x`` ``>`` ``y` operator on JAX arrays.

- [`jax.lax.eq()`](jax.lax.eq.html#jax.lax.eq "jax.lax.eq"): elementwise equal

- [`jax.lax.ne()`](jax.lax.ne.html#jax.lax.ne "jax.lax.ne"): elementwise not-equal

- [`jax.lax.ge()`](jax.lax.ge.html#jax.lax.ge "jax.lax.ge"): elementwise greater-than-or-equal

- [`jax.lax.le()`](jax.lax.le.html#jax.lax.le "jax.lax.le"): elementwise less-than-or-equal

- [`jax.lax.lt()`](jax.lax.lt.html#jax.lax.lt "jax.lax.lt"): elementwise less-than

[](jax.lax.ge.html "previous page")

previous

jax.lax.ge

[](jax.lax.igamma.html "next page")

next

jax.lax.igamma

Contents

- [`gt()`](#jax.lax.gt)

By The JAX authors

© Copyright 2024, The JAX Authors.\
