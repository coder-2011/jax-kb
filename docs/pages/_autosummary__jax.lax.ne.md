- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.ne

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.ne.rst "Download source file")
-  .pdf

# jax.lax.ne

## Contents

- [`ne()`](#jax.lax.ne)

# jax.lax.ne[\#](#jax-lax-ne "Link to this heading")

jax.lax.ne(*x*, *y*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L1463-L1493)[\#](#jax.lax.ne "Link to this definition")  
Elementwise not-equals: \\x \neq y\\.

This function lowers directly to the [stablehlo.compare](https://openxla.org/stablehlo/spec#compare) operation with `comparison_direction=NE` and `compare_type` set according to the input dtype.

Parameters:  
- **x** (*ArrayLike*) – Input arrays. Must have matching dtypes. If neither is a scalar, `x` and `y` must have the same number of dimensions and be broadcast compatible.

- **y** (*ArrayLike*) – Input arrays. Must have matching dtypes. If neither is a scalar, `x` and `y` must have the same number of dimensions and be broadcast compatible.

Returns:  
A boolean array of shape `lax.broadcast_shapes(x.shape,`` ``y.shape)` containing the elementwise not-equal comparison.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.not_equal()`](jax.numpy.not_equal.html#jax.numpy.not_equal "jax.numpy.not_equal"): NumPy wrapper for this API, also accessible via the `x`` ``!=`` ``y` operator on JAX arrays.

- [`jax.lax.eq()`](jax.lax.eq.html#jax.lax.eq "jax.lax.eq"): elementwise equal

- [`jax.lax.ge()`](jax.lax.ge.html#jax.lax.ge "jax.lax.ge"): elementwise greater-than-or-equal

- [`jax.lax.gt()`](jax.lax.gt.html#jax.lax.gt "jax.lax.gt"): elementwise greater-than

- [`jax.lax.le()`](jax.lax.le.html#jax.lax.le "jax.lax.le"): elementwise less-than-or-equal

- [`jax.lax.lt()`](jax.lax.lt.html#jax.lax.lt "jax.lax.lt"): elementwise less-than

[](jax.lax.mulhi.html "previous page")

previous

jax.lax.mulhi

[](jax.lax.neg.html "next page")

next

jax.lax.neg

Contents

- [`ne()`](#jax.lax.ne)

By The JAX authors

© Copyright 2024, The JAX Authors.\
