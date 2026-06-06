- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.tpu` module](../jax.experimental.pallas.tpu.html)
- jax.experimental.pallas.tpu.prng_seed

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.tpu.prng_seed.rst "Download source file")
-  .pdf

# jax.experimental.pallas.tpu.prng_seed

## Contents

- [`prng_seed()`](#jax.experimental.pallas.tpu.prng_seed)

# jax.experimental.pallas.tpu.prng_seed[\#](#jax-experimental-pallas-tpu-prng-seed "Link to this heading")

jax.experimental.pallas.tpu.prng_seed(*\*seeds*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic/primitives.py#L815-L824)[\#](#jax.experimental.pallas.tpu.prng_seed "Link to this definition")  
Sets the seed for PRNG.

Parameters:  
**seeds** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*jax.Array*](jax.Array.html#jax.Array "jax.Array")) – One or more integer seeds for setting the PRNG seed. If more than one seed is passed in, the seed material will be mixed before setting the internal PRNG state.

Return type:  
None

[](jax.experimental.pallas.tpu.emit_pipeline_with_allocations.html "previous page")

previous

jax.experimental.pallas.tpu.emit_pipeline_with_allocations

[](jax.experimental.pallas.tpu.sample_block.html "next page")

next

jax.experimental.pallas.tpu.sample_block

Contents

- [`prng_seed()`](#jax.experimental.pallas.tpu.prng_seed)

By The JAX authors

© Copyright 2024, The JAX Authors.\
