- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.clz

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.clz.rst "Download source file")
-  .pdf

# jax.lax.clz

## Contents

- [`clz()`](#jax.lax.clz)

# jax.lax.clz[\#](#jax-lax-clz "Link to this heading")

jax.lax.clz(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L1122-L1141)[\#](#jax.lax.clz "Link to this definition")  
Elementwise count-leading-zeros.

This function lowers directly to the [stablehlo.count_leading_zeros](https://openxla.org/stablehlo/spec#count_leading_zeros) operation.

Parameters:  
**x** (*ArrayLike*) – Input array. Must have integer dtype.

Returns:  
An array of the same shape and dtype as `x`, containing the number of leading zeros in the input.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.lax.population_count()`](jax.lax.population_count.html#jax.lax.population_count "jax.lax.population_count"): Count the number of set bits in each element.

[](jax.lax.clamp.html "previous page")

previous

jax.lax.clamp

[](jax.lax.collapse.html "next page")

next

jax.lax.collapse

Contents

- [`clz()`](#jax.lax.clz)

By The JAX authors

© Copyright 2024, The JAX Authors.\
