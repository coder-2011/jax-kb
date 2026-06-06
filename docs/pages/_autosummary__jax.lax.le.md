- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.le

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.le.rst "Download source file")
-  .pdf

# jax.lax.le

## Contents

- [`le()`](#jax.lax.le)

# jax.lax.le[\#](#jax-lax-le "Link to this heading")

jax.lax.le(*x*, *y*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L1556-L1586)[\#](#jax.lax.le "Link to this definition")  
Elementwise less-than-or-equals: \\x \leq y\\.

This function lowers directly to the [stablehlo.compare](https://openxla.org/stablehlo/spec#compare) operation with `comparison_direction=LE` and `compare_type` set according to the input dtype.

Parameters:  
- **x** (*ArrayLike*) – Input arrays. Must have matching non-complex dtypes. If neither is a scalar, `x` and `y` must have the same number of dimensions and be broadcast compatible.

- **y** (*ArrayLike*) – Input arrays. Must have matching non-complex dtypes. If neither is a scalar, `x` and `y` must have the same number of dimensions and be broadcast compatible.

Returns:  
A boolean array of shape `lax.broadcast_shapes(x.shape,`` ``y.shape)` containing the elementwise less-than-or-equal comparison.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.less_equal()`](jax.numpy.less_equal.html#jax.numpy.less_equal "jax.numpy.less_equal"): NumPy wrapper for this API, also accessible via the `x`` ``<=`` ``y` operator on JAX arrays.

- [`jax.lax.eq()`](jax.lax.eq.html#jax.lax.eq "jax.lax.eq"): elementwise equal

- [`jax.lax.ne()`](jax.lax.ne.html#jax.lax.ne "jax.lax.ne"): elementwise not-equal

- [`jax.lax.ge()`](jax.lax.ge.html#jax.lax.ge "jax.lax.ge"): elementwise greater-than-or-equal

- [`jax.lax.gt()`](jax.lax.gt.html#jax.lax.gt "jax.lax.gt"): elementwise greater-than

- [`jax.lax.lt()`](jax.lax.lt.html#jax.lax.lt "jax.lax.lt"): elementwise less-than

[](jax.lax.is_finite.html "previous page")

previous

jax.lax.is_finite

[](jax.lax.lgamma.html "next page")

next

jax.lax.lgamma

Contents

- [`le()`](#jax.lax.le)

By The JAX authors

© Copyright 2024, The JAX Authors.\
