- [](../index.html)
- [API Reference](../jax.html)
- [`jax.random` module](../jax.random.html)
- jax.random.poisson

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.random.poisson.rst "Download source file")
-  .pdf

# jax.random.poisson

## Contents

- [`poisson()`](#jax.random.poisson)

# jax.random.poisson[\#](#jax-random-poisson "Link to this heading")

jax.random.poisson(*key*, *lam*, *shape=None*, *dtype=None*, *\**, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/random/core.py#L1836-L1890)[\#](#jax.random.poisson "Link to this definition")  
Sample Poisson random values with given shape and integer dtype.

The values are distributed according to the probability mass function:

\\f(k; \lambda) = \frac{\lambda^k e^{-\lambda}}{k!}\\

Where k is a non-negative integer and \\\lambda \> 0\\.

Parameters:  
- **key** (*ArrayLike*) – a PRNG key used as the random key.

- **lam** (*RealArray*) – rate parameter (mean of the distribution), must be \>= 0. Must be broadcast-compatible with `shape`

- **shape** (*Shape* *\|* *None*) – optional, a tuple of nonnegative integers representing the result shape. Default (None) produces a result shape equal to `lam.shape`.

- **dtype** (*DTypeLikeInt* *\|* *None*) – optional, a integer dtype for the returned values (default int64 if jax_enable_x64 is true, otherwise int32).

- **out_sharding** ([*NamedSharding*](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding") *\|* *P* *\|* *None*) – Optional. Specifies how the output array should be sharded across devices in multi-device computation. Can be a [`NamedSharding`](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding"), a [`PartitionSpec`](../jax.sharding.html#jax.sharding.PartitionSpec "jax.sharding.PartitionSpec") (`P`), or `None` (default). When specified, the output will be sharded according to the given sharding specification. Primarily used in explicit sharding mode. See the [explicit sharding tutorial](https://docs.jax.dev/en/latest/parallel.html) for more details.

Returns:  
A random array with the specified dtype and with shape given by `shape` if `shape`` ``is`` ``not`` ``None,`` ``or`` ``else`` ``by`` ```` ``lam.shape ```.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.random.permutation.html "previous page")

previous

jax.random.permutation

[](jax.random.rademacher.html "next page")

next

jax.random.rademacher

Contents

- [`poisson()`](#jax.random.poisson)

By The JAX authors

© Copyright 2024, The JAX Authors.\
