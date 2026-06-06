- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.integer_pow

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.integer_pow.rst "Download source file")
-  .pdf

# jax.lax.integer_pow

## Contents

- [`integer_pow()`](#jax.lax.integer_pow)

# jax.lax.integer_pow[\#](#jax-lax-integer-pow "Link to this heading")

jax.lax.integer_pow(*x*, *y*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L892-L912)[\#](#jax.lax.integer_pow "Link to this definition")  
Elementwise power: \\x^y\\, where \\y\\ is a static integer.

This will lower to a sequence of \\O\[\log_2(y)\]\\ repetitions of [stablehlo.multiply](https://openxla.org/stablehlo/spec#multiply).

Parameters:  
- **x** (*ArrayLike*) – Input array giving the base value. Must have numerical dtype.

- **y** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – Static scalar integer giving the exponent.

Returns:  
An array of the same shape and dtype as `x` containing the elementwise power.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

[`jax.lax.pow()`](jax.lax.pow.html#jax.lax.pow "jax.lax.pow"): Elementwise power where `y` is an array.

[](jax.lax.index_take.html "previous page")

previous

jax.lax.index_take

[](jax.lax.iota.html "next page")

next

jax.lax.iota

Contents

- [`integer_pow()`](#jax.lax.integer_pow)

By The JAX authors

© Copyright 2024, The JAX Authors.\
