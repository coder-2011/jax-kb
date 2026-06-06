- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.tpu` module](../jax.experimental.pallas.tpu.html)
- jax.experimental.pallas.tpu.sample_block

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.tpu.sample_block.rst "Download source file")
-  .pdf

# jax.experimental.pallas.tpu.sample_block

## Contents

- [`sample_block()`](#jax.experimental.pallas.tpu.sample_block)

# jax.experimental.pallas.tpu.sample_block[\#](#jax-experimental-pallas-tpu-sample-block "Link to this heading")

jax.experimental.pallas.tpu.sample_block(*sampler_fn*, *global_key*, *block_size*, *tile_size*, *total_size*, *block_index=None*, *\*\*kwargs*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic/random.py#L166-L215)[\#](#jax.experimental.pallas.tpu.sample_block "Link to this definition")  
Samples a block of random values with invariance guarantees.

`sample_block` allows the sampling of identical blocks of random values across kernels with different block shapes and iteration orders. Each call to sample_block returns a block_size-shaped array of random samples corresponding to the block_index.

`tile_size` should be chosen such that it is a divisor to all block sizes one needs to be invariant to. The larger the `tile_size`, the more efficient the sampling process will be and therefore the best choice is typically the greatest common divisor between all possible block sizes.

Parameters:  
- **sampler_fn** (*SampleFn*) – A sampling function that consumes a key and returns random samples.

- **global_key** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – The global key to use for sampling.

- **block_size** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *...\]*) – The shape of an individual block.

- **tile_size** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *...\]*) – The shape of a `tile`, which is the smallest unit at which samples are generated. This should be selected to be a divisor of all block sizes one needs to be invariant to.

- **total_size** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *...\]*) – The total size of the array to sample.

- **block_index** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")*,* *...\]* *\|* *None*) – The index denoting which block to generate keys for. Defaults to the program_id for each block axis.

- **\*\*kwargs** – Additional arguments to pass to the sampler_fn.

Returns:  
A `block_size` shaped array of samples for the current block corresponding to `block_index`.

Return type:  
[*Array*](jax.Array.html#jax.Array "jax.Array")

[](jax.experimental.pallas.tpu.prng_seed.html "previous page")

previous

jax.experimental.pallas.tpu.prng_seed

[](jax.experimental.pallas.tpu.stateful_bernoulli.html "next page")

next

jax.experimental.pallas.tpu.stateful_bernoulli

Contents

- [`sample_block()`](#jax.experimental.pallas.tpu.sample_block)

By The JAX authors

© Copyright 2024, The JAX Authors.\
