- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.rng_bit_generator

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.rng_bit_generator.rst "Download source file")
-  .pdf

# jax.lax.rng_bit_generator

## Contents

- [`rng_bit_generator()`](#jax.lax.rng_bit_generator)

# jax.lax.rng_bit_generator[\#](#jax-lax-rng-bit-generator "Link to this heading")

jax.lax.rng_bit_generator(*key*, *shape*, *dtype=\<class 'numpy.uint32'\>*, *algorithm=RandomAlgorithm.RNG_DEFAULT*, *\**, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L9225-L9251)[\#](#jax.lax.rng_bit_generator "Link to this definition")  
Stateless PRNG bit generator. Experimental and its use is discouraged.

Returns uniformly distributed random bits with the specified shape and dtype (what is required to be an integer type) using the platform specific default algorithm or the one specified.

It provides direct access to the RngBitGenerator primitive exposed by XLA ([https://www.openxla.org/xla/operation_semantics#rngbitgenerator](https://www.openxla.org/xla/operation_semantics#rngbitgenerator)) for low level API access.

Most users should use jax.random instead for a stable and more user friendly API.

[](jax.lax.rev.html "previous page")

previous

jax.lax.rev

[](jax.lax.rng_uniform.html "next page")

next

jax.lax.rng_uniform

Contents

- [`rng_bit_generator()`](#jax.lax.rng_bit_generator)

By The JAX authors

© Copyright 2024, The JAX Authors.\
