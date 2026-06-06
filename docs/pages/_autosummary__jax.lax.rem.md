- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.rem

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.rem.rst "Download source file")
-  .pdf

# jax.lax.rem

## Contents

- [`rem()`](#jax.lax.rem)

# jax.lax.rem[\#](#jax-lax-rem "Link to this heading")

jax.lax.rem(*x*, *y*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L1272-L1299)[\#](#jax.lax.rem "Link to this definition")  
Elementwise remainder: \\x \bmod y\\.

This function lowers directly to the [stablehlo.remainder](https://openxla.org/stablehlo/spec#remainder) operation. The sign of the result is taken from the dividend, and the absolute value of the result is always less than the divisor’s absolute value.

Integer division overflow (remainder by zero or remainder of INT_SMIN with -1) produces an implementation defined value.

Parameters:  
- **x** (*ArrayLike*) – Input arrays. Must have matching int or float dtypes. If neither is a scalar, `x` and `y` must have the same number of dimensions and be broadcast compatible.

- **y** (*ArrayLike*) – Input arrays. Must have matching int or float dtypes. If neither is a scalar, `x` and `y` must have the same number of dimensions and be broadcast compatible.

Returns:  
An array of the same dtype as `x` and `y` containing the remainder.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.remainder()`](jax.numpy.remainder.html#jax.numpy.remainder "jax.numpy.remainder"): NumPy-style remainder with different sign semantics.

[](jax.lax.reduce_xor.html "previous page")

previous

jax.lax.reduce_xor

[](jax.lax.reshape.html "next page")

next

jax.lax.reshape

Contents

- [`rem()`](#jax.lax.rem)

By The JAX authors

© Copyright 2024, The JAX Authors.\
