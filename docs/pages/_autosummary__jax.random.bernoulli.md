- [](../index.html)
- [API Reference](../jax.html)
- [`jax.random` module](../jax.random.html)
- jax.random.bernoulli

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.random.bernoulli.rst "Download source file")
-  .pdf

# jax.random.bernoulli

## Contents

- [`bernoulli()`](#jax.random.bernoulli)

# jax.random.bernoulli[\#](#jax-random-bernoulli "Link to this heading")

jax.random.bernoulli(*key*, *p=0.5*, *shape=None*, *mode='low'*, *\**, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/random/core.py#L1151-L1204)[\#](#jax.random.bernoulli "Link to this definition")  
Sample Bernoulli random values with given shape and mean.

The values are distributed according to the probability mass function:

\\f(k; p) = p^k(1 - p)^{1 - k}\\

where \\k \in \\0, 1\\\\ and \\0 \le p \le 1\\.

Parameters:  
- **key** (*ArrayLike*) – a PRNG key used as the random key.

- **p** (*RealArray*) – optional, a float or array of floats for the mean of the random variables. Must be broadcast-compatible with `shape`. Default 0.5.

- **shape** (*Shape* *\|* *None*) – optional, a tuple of nonnegative integers representing the result shape. Must be broadcast-compatible with `p.shape`. The default (None) produces a result shape equal to `p.shape`.

- **mode** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – optional, “high” or “low” for how many bits to use when sampling. default=’low’. Set to “high” for correct sampling at small values of p. When sampling in float32, bernoulli samples with mode=’low’ produce incorrect results for p \< ~1E-7. mode=”high” approximately doubles the cost of sampling.

- **out_sharding** ([*NamedSharding*](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding") *\|* *P* *\|* *None*) – Optional. Specifies how the output array should be sharded across devices in multi-device computation. Can be a [`NamedSharding`](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding"), a [`PartitionSpec`](../jax.sharding.html#jax.sharding.PartitionSpec "jax.sharding.PartitionSpec") (`P`), or `None` (default). When specified, the output will be sharded according to the given sharding specification. Primarily used in explicit sharding mode. See the [explicit sharding tutorial](https://docs.jax.dev/en/latest/parallel.html) for more details.

Returns:  
A random array with boolean dtype and shape given by `shape` if `shape` is not None, or else `p.shape`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.random.ball.html "previous page")

previous

jax.random.ball

[](jax.random.beta.html "next page")

next

jax.random.beta

Contents

- [`bernoulli()`](#jax.random.bernoulli)

By The JAX authors

© Copyright 2024, The JAX Authors.\
