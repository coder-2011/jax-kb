- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.tpu` module](../jax.experimental.pallas.tpu.html)
- jax.experimental.pallas.tpu.stateful_bernoulli

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.tpu.stateful_bernoulli.rst "Download source file")
-  .pdf

# jax.experimental.pallas.tpu.stateful_bernoulli

## Contents

- [`stateful_bernoulli()`](#jax.experimental.pallas.tpu.stateful_bernoulli)

# jax.experimental.pallas.tpu.stateful_bernoulli[\#](#jax-experimental-pallas-tpu-stateful-bernoulli "Link to this heading")

jax.experimental.pallas.tpu.stateful_bernoulli(*\*args*, *\*\*kwargs*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic/random.py#L147-L153)[\#](#jax.experimental.pallas.tpu.stateful_bernoulli "Link to this definition")  
Sample Bernoulli random values with given shape and mean.

The values are distributed according to the probability mass function:

\\f(k; p) = p^k(1 - p)^{1 - k}\\

where \\k \in \\0, 1\\\\ and \\0 \le p \le 1\\.

Parameters:  
- **p** – optional, a float or array of floats for the mean of the random variables. Must be broadcast-compatible with `shape`. Default 0.5.

- **shape** – optional, a tuple of nonnegative integers representing the result shape. Must be broadcast-compatible with `p.shape`. The default (None) produces a result shape equal to `p.shape`.

- **mode** – optional, “high” or “low” for how many bits to use when sampling. default=’low’. Set to “high” for correct sampling at small values of p. When sampling in float32, bernoulli samples with mode=’low’ produce incorrect results for p \< ~1E-7. mode=”high” approximately doubles the cost of sampling.

- **out_sharding** – Optional. Specifies how the output array should be sharded across devices in multi-device computation. Can be a [`NamedSharding`](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding"), a [`PartitionSpec`](../jax.sharding.html#jax.sharding.PartitionSpec "jax.sharding.PartitionSpec") (`P`), or `None` (default). When specified, the output will be sharded according to the given sharding specification. Primarily used in explicit sharding mode. See the [explicit sharding tutorial](https://docs.jax.dev/en/latest/parallel.html) for more details.

Returns:  
A random array with boolean dtype and shape given by `shape` if `shape` is not None, or else `p.shape`.

[](jax.experimental.pallas.tpu.sample_block.html "previous page")

previous

jax.experimental.pallas.tpu.sample_block

[](jax.experimental.pallas.tpu.stateful_bits.html "next page")

next

jax.experimental.pallas.tpu.stateful_bits

Contents

- [`stateful_bernoulli()`](#jax.experimental.pallas.tpu.stateful_bernoulli)

By The JAX authors

© Copyright 2024, The JAX Authors.\
