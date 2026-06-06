- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.eq

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.eq.rst "Download source file")
-  .pdf

# jax.lax.eq

## Contents

- [`eq()`](#jax.lax.eq)

# jax.lax.eq[\#](#jax-lax-eq "Link to this heading")

jax.lax.eq(*x*, *y*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L1432-L1462)[\#](#jax.lax.eq "Link to this definition")  
Elementwise equals: \\x = y\\.

This function lowers directly to the [stablehlo.compare](https://openxla.org/stablehlo/spec#compare) operation with `comparison_direction=EQ` and `compare_type` set according to the input dtype.

Parameters:  
- **x** (*ArrayLike*) – Input arrays. Must have matching dtypes. If neither is a scalar, `x` and `y` must have the same number of dimensions and be broadcast compatible.

- **y** (*ArrayLike*) – Input arrays. Must have matching dtypes. If neither is a scalar, `x` and `y` must have the same number of dimensions and be broadcast compatible.

Returns:  
A boolean array of shape `lax.broadcast_shapes(x.shape,`` ``y.shape)` containing the elementwise equal comparison.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.equal()`](jax.numpy.equal.html#jax.numpy.equal "jax.numpy.equal"): NumPy wrapper for this API, also accessible via the `x`` ``==`` ``y` operator on JAX arrays.

- [`jax.lax.ne()`](jax.lax.ne.html#jax.lax.ne "jax.lax.ne"): elementwise not-equal

- [`jax.lax.ge()`](jax.lax.ge.html#jax.lax.ge "jax.lax.ge"): elementwise greater-than-or-equal

- [`jax.lax.gt()`](jax.lax.gt.html#jax.lax.gt "jax.lax.gt"): elementwise greater-than

- [`jax.lax.le()`](jax.lax.le.html#jax.lax.le "jax.lax.le"): elementwise less-than-or-equal

- [`jax.lax.lt()`](jax.lax.lt.html#jax.lax.lt "jax.lax.lt"): elementwise less-than

[](jax.lax.empty.html "previous page")

previous

jax.lax.empty

[](jax.lax.erf.html "next page")

next

jax.lax.erf

Contents

- [`eq()`](#jax.lax.eq)

By The JAX authors

© Copyright 2024, The JAX Authors.\
