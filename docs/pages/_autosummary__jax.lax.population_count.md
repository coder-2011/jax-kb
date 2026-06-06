- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.population_count

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.population_count.rst "Download source file")
-  .pdf

# jax.lax.population_count

## Contents

- [`population_count()`](#jax.lax.population_count)

# jax.lax.population_count[\#](#jax-lax-population-count "Link to this heading")

jax.lax.population_count(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L1101-L1121)[\#](#jax.lax.population_count "Link to this definition")  
Elementwise popcount, count the number of set bits in each element.

This function lowers directly to the [stablehlo.popcnt](https://openxla.org/stablehlo/spec#popcnt) operation.

Parameters:  
**x** (*ArrayLike*) – Input array. Must have integer dtype.

Returns:  
An array of the same shape and dtype as `x`, containing the number of set bits in the input.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.lax.clz()`](jax.lax.clz.html#jax.lax.clz "jax.lax.clz"): Elementwise count leading zeros.

- [`jax.numpy.bitwise_count()`](jax.numpy.bitwise_count.html#jax.numpy.bitwise_count "jax.numpy.bitwise_count"): More flexible NumPy-style API for bit counts.

[](jax.lax.bitwise_xor.html "previous page")

previous

jax.lax.bitwise_xor

[](jax.lax.broadcast.html "next page")

next

jax.lax.broadcast

Contents

- [`population_count()`](#jax.lax.population_count)

By The JAX authors

© Copyright 2024, The JAX Authors.\
