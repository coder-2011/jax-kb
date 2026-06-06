- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.tpu` module](../jax.experimental.pallas.tpu.html)
- jax.experimental.pallas.tpu.stateful_bits

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.tpu.stateful_bits.rst "Download source file")
-  .pdf

# jax.experimental.pallas.tpu.stateful_bits

## Contents

- [`stateful_bits()`](#jax.experimental.pallas.tpu.stateful_bits)

# jax.experimental.pallas.tpu.stateful_bits[\#](#jax-experimental-pallas-tpu-stateful-bits "Link to this heading")

jax.experimental.pallas.tpu.stateful_bits(*\*args*, *\*\*kwargs*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic/random.py#L147-L153)[\#](#jax.experimental.pallas.tpu.stateful_bits "Link to this definition")  
Sample uniform bits in the form of unsigned integers.

Parameters:  
- **shape** – optional, a tuple of nonnegative integers representing the result shape. Default `()`.

- **dtype** – optional, an unsigned integer dtype for the returned values (default `uint64` if `jax_enable_x64` is true, otherwise `uint32`).

- **out_sharding** – Optional. Specifies how the output array should be sharded across devices in multi-device computation. Can be a [`NamedSharding`](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding"), a [`PartitionSpec`](../jax.sharding.html#jax.sharding.PartitionSpec "jax.sharding.PartitionSpec") (`P`), or `None` (default). When specified, the output will be sharded according to the given sharding specification. Primarily used in explicit sharding mode. See the [explicit sharding tutorial](https://docs.jax.dev/en/latest/parallel.html) for more details.

Returns:  
A random array with the specified shape and dtype.

[](jax.experimental.pallas.tpu.stateful_bernoulli.html "previous page")

previous

jax.experimental.pallas.tpu.stateful_bernoulli

[](jax.experimental.pallas.tpu.stateful_normal.html "next page")

next

jax.experimental.pallas.tpu.stateful_normal

Contents

- [`stateful_bits()`](#jax.experimental.pallas.tpu.stateful_bits)

By The JAX authors

© Copyright 2024, The JAX Authors.\
