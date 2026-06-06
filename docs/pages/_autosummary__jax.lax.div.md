- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.div

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.div.rst "Download source file")
-  .pdf

# jax.lax.div

## Contents

- [`div()`](#jax.lax.div)

# jax.lax.div[\#](#jax-lax-div "Link to this heading")

jax.lax.div(*x*, *y*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L1242-L1271)[\#](#jax.lax.div "Link to this definition")  
Elementwise division: \\x \over y\\.

This function lowers directly to the [stablehlo.divide](https://openxla.org/stablehlo/spec#divide) operation.

Integer division overflow (division by zero or signed division of INT_SMIN with -1) produces an implementation defined value.

Parameters:  
- **x** (*ArrayLike*) – Input arrays. Must have matching numerical dtypes. If neither is a scalar, `x` and `y` must have the same number of dimensions and be broadcast compatible.

- **y** (*ArrayLike*) – Input arrays. Must have matching numerical dtypes. If neither is a scalar, `x` and `y` must have the same number of dimensions and be broadcast compatible.

Returns:  
An array of the same dtype as `x` and `y` containing the quotient of each pair of broadcasted entries. For integer inputs, any fractional part is discarded.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.divide()`](jax.numpy.divide.html#jax.numpy.divide "jax.numpy.divide"): NumPy-style true division supporting inputs with mixed dtypes and ranks.

- [`jax.numpy.floor_divide()`](jax.numpy.floor_divide.html#jax.numpy.floor_divide "jax.numpy.floor_divide"): NumPy-style floor division supporting inputs with mixed dtypes and ranks.

[](jax.lax.digamma.html "previous page")

previous

jax.lax.digamma

[](jax.lax.dot.html "next page")

next

jax.lax.dot

Contents

- [`div()`](#jax.lax.div)

By The JAX authors

© Copyright 2024, The JAX Authors.\
