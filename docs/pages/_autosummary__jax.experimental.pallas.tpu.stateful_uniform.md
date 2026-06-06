- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.tpu` module](../jax.experimental.pallas.tpu.html)
- jax.experimental.pallas.tpu.stateful_uniform

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.tpu.stateful_uniform.rst "Download source file")
-  .pdf

# jax.experimental.pallas.tpu.stateful_uniform

## Contents

- [`stateful_uniform()`](#jax.experimental.pallas.tpu.stateful_uniform)

# jax.experimental.pallas.tpu.stateful_uniform[\#](#jax-experimental-pallas-tpu-stateful-uniform "Link to this heading")

jax.experimental.pallas.tpu.stateful_uniform(*\*args*, *\*\*kwargs*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic/random.py#L147-L153)[\#](#jax.experimental.pallas.tpu.stateful_uniform "Link to this definition")  
Sample uniform random values in \[minval, maxval) with given shape/dtype.

Parameters:  
- **shape** – optional, a tuple of nonnegative integers representing the result shape. Default ().

- **dtype** – optional, a float dtype for the returned values (default float64 if jax_enable_x64 is true, otherwise float32).

- **minval** – optional, a minimum (inclusive) value broadcast-compatible with shape for the range (default 0).

- **maxval** – optional, a maximum (exclusive) value broadcast-compatible with shape for the range (default 1).

- **out_sharding** – Optional. Specifies how the output array should be sharded across devices in multi-device computation. Can be a [`NamedSharding`](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding"), a [`PartitionSpec`](../jax.sharding.html#jax.sharding.PartitionSpec "jax.sharding.PartitionSpec") (`P`), or `None` (default). When specified, the output will be sharded according to the given sharding specification. Primarily used in explicit sharding mode. See the [explicit sharding tutorial](https://docs.jax.dev/en/latest/parallel.html) for more details.

Returns:  
A random array with the specified shape and dtype.

[](jax.experimental.pallas.tpu.stateful_normal.html "previous page")

previous

jax.experimental.pallas.tpu.stateful_normal

[](jax.experimental.pallas.tpu.to_pallas_key.html "next page")

next

jax.experimental.pallas.tpu.to_pallas_key

Contents

- [`stateful_uniform()`](#jax.experimental.pallas.tpu.stateful_uniform)

By The JAX authors

© Copyright 2024, The JAX Authors.\
