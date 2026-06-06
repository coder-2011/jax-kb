- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.pow

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.pow.rst "Download source file")
-  .pdf

# jax.lax.pow

## Contents

- [`pow()`](#jax.lax.pow)

# jax.lax.pow[\#](#jax-lax-pow "Link to this heading")

jax.lax.pow(*x*, *y*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L866-L891)[\#](#jax.lax.pow "Link to this definition")  
Elementwise power: \\x^y\\.

This function lowers directly to the [stablehlo.pow](https://openxla.org/stablehlo/spec#pow) operation, along with a [stablehlo.convert](https://openxla.org/stablehlo/spec#convert) when the argument dtypes do not match.

Parameters:  
- **x** (*ArrayLike*) – Input array giving the base value. Must have floating or complex type.

- **y** (*ArrayLike*) – Input array giving the exponent value. Must have integer, floating, or complex type. Its dtype will be cast to that of `x.dtype` if necessary. If neither `x` nor `y` is a scalar, then `x` and `y` must have the same number of dimensions and be broadcast-compatible.

Returns:  
An array of the same dtype as `x` containing the elementwise power.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

[`jax.lax.integer_pow()`](jax.lax.integer_pow.html#jax.lax.integer_pow "jax.lax.integer_pow"): Elementwise power where `y` is a static integer.

[](jax.lax.polygamma.html "previous page")

previous

jax.lax.polygamma

[](jax.lax.random_gamma_grad.html "next page")

next

jax.lax.random_gamma_grad

Contents

- [`pow()`](#jax.lax.pow)

By The JAX authors

© Copyright 2024, The JAX Authors.\
