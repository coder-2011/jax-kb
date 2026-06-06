- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.ge

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.ge.rst "Download source file")
-  .pdf

# jax.lax.ge

## Contents

- [`ge()`](#jax.lax.ge)

# jax.lax.ge[\#](#jax-lax-ge "Link to this heading")

jax.lax.ge(*x*, *y*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L1494-L1524)[\#](#jax.lax.ge "Link to this definition")  
Elementwise greater-than-or-equals: \\x \geq y\\.

This function lowers directly to the [stablehlo.compare](https://openxla.org/stablehlo/spec#compare) operation with `comparison_direction=GE` and `compare_type` set according to the input dtype.

Parameters:  
- **x** (*ArrayLike*) – Input arrays. Must have matching non-complex dtypes. If neither is a scalar, `x` and `y` must have the same number of dimensions and be broadcast compatible.

- **y** (*ArrayLike*) – Input arrays. Must have matching non-complex dtypes. If neither is a scalar, `x` and `y` must have the same number of dimensions and be broadcast compatible.

Returns:  
A boolean array of shape `lax.broadcast_shapes(x.shape,`` ``y.shape)` containing the elementwise greater-than-or-equal comparison.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.greater_equal()`](jax.numpy.greater_equal.html#jax.numpy.greater_equal "jax.numpy.greater_equal"): NumPy wrapper for this API, also accessible via the `x`` ``>=`` ``y` operator on JAX arrays.

- [`jax.lax.eq()`](jax.lax.eq.html#jax.lax.eq "jax.lax.eq"): elementwise equal

- [`jax.lax.ne()`](jax.lax.ne.html#jax.lax.ne "jax.lax.ne"): elementwise not-equal

- [`jax.lax.gt()`](jax.lax.gt.html#jax.lax.gt "jax.lax.gt"): elementwise greater-than

- [`jax.lax.le()`](jax.lax.le.html#jax.lax.le "jax.lax.le"): elementwise less-than-or-equal

- [`jax.lax.lt()`](jax.lax.lt.html#jax.lax.lt "jax.lax.lt"): elementwise less-than

[](jax.lax.gather.html "previous page")

previous

jax.lax.gather

[](jax.lax.gt.html "next page")

next

jax.lax.gt

Contents

- [`ge()`](#jax.lax.ge)

By The JAX authors

© Copyright 2024, The JAX Authors.\
