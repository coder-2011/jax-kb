- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.rng_uniform

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.rng_uniform.rst "Download source file")
-  .pdf

# jax.lax.rng_uniform

## Contents

- [`rng_uniform()`](#jax.lax.rng_uniform)

# jax.lax.rng_uniform[\#](#jax-lax-rng-uniform "Link to this heading")

jax.lax.rng_uniform(*a*, *b*, *shape*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L9014-L9028)[\#](#jax.lax.rng_uniform "Link to this definition")  
Stateful PRNG generator. Experimental and its use is discouraged.

Returns uniformly distributed random numbers in the range \[a, b). If b \<= a, then the result is undefined, and different implementations may return different results.

You should use jax.random for most purposes; this function exists only for niche use cases with special performance requirements.

This API may be removed at any time.

[](jax.lax.rng_bit_generator.html "previous page")

previous

jax.lax.rng_bit_generator

[](jax.lax.round.html "next page")

next

jax.lax.round

Contents

- [`rng_uniform()`](#jax.lax.rng_uniform)

By The JAX authors

© Copyright 2024, The JAX Authors.\
