- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.tpu` module](../jax.experimental.pallas.tpu.html)
- jax.experimental.pallas.tpu.stateful_normal

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.tpu.stateful_normal.rst "Download source file")
-  .pdf

# jax.experimental.pallas.tpu.stateful_normal

## Contents

- [`stateful_normal()`](#jax.experimental.pallas.tpu.stateful_normal)

# jax.experimental.pallas.tpu.stateful_normal[\#](#jax-experimental-pallas-tpu-stateful-normal "Link to this heading")

jax.experimental.pallas.tpu.stateful_normal(*\*args*, *\*\*kwargs*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic/random.py#L147-L153)[\#](#jax.experimental.pallas.tpu.stateful_normal "Link to this definition")  
Sample standard normal random values with given shape and float dtype.

The values are returned according to the probability density function:

\\f(x) = \frac{1}{\sqrt{2\pi}}e^{-x^2/2}\\

on the domain \\-\infty \< x \< \infty\\

Parameters:  
- **shape** – optional, a tuple of nonnegative integers representing the result shape. Default ().

- **dtype** – optional, a float dtype for the returned values (default float64 if jax_enable_x64 is true, otherwise float32).

- **out_sharding** – Optional. Specifies how the output array should be sharded across devices in multi-device computation. Can be a [`NamedSharding`](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding"), a [`PartitionSpec`](../jax.sharding.html#jax.sharding.PartitionSpec "jax.sharding.PartitionSpec") (`P`), or `None` (default). When specified, the output will be sharded according to the given sharding specification. Primarily used in explicit sharding mode. See the [explicit sharding tutorial](https://docs.jax.dev/en/latest/parallel.html) for more details.

Returns:  
A random array with the specified shape and dtype.

[](jax.experimental.pallas.tpu.stateful_bits.html "previous page")

previous

jax.experimental.pallas.tpu.stateful_bits

[](jax.experimental.pallas.tpu.stateful_uniform.html "next page")

next

jax.experimental.pallas.tpu.stateful_uniform

Contents

- [`stateful_normal()`](#jax.experimental.pallas.tpu.stateful_normal)

By The JAX authors

© Copyright 2024, The JAX Authors.\
