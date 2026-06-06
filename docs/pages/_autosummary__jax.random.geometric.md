- [](../index.html)
- [API Reference](../jax.html)
- [`jax.random` module](../jax.random.html)
- jax.random.geometric

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.random.geometric.rst "Download source file")
-  .pdf

# jax.random.geometric

## Contents

- [`geometric()`](#jax.random.geometric)

# jax.random.geometric[\#](#jax-random-geometric "Link to this heading")

jax.random.geometric(*key*, *p*, *shape=None*, *dtype=None*, *\**, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/random/core.py#L2937-L2983)[\#](#jax.random.geometric "Link to this definition")  
Sample Geometric random values with given shape and float dtype.

The values are returned according to the probability mass function:

\\f(k;p) = p(1-p)^{k-1}\\

on the domain \\0 \< p \< 1\\.

Parameters:  
- **key** (*ArrayLike*) – a PRNG key used as the random key.

- **p** (*RealArray*) – a float or array of floats broadcast-compatible with `shape` representing the probability of success of an individual trial.

- **shape** (*Shape* *\|* *None*) – optional, a tuple of nonnegative integers specifying the result shape. Must be broadcast-compatible with `p`. The default (None) produces a result shape equal to `np.shape(p)`.

- **dtype** (*DTypeLikeInt* *\|* *None*) – optional, a int dtype for the returned values (default int64 if jax_enable_x64 is true, otherwise int32).

- **out_sharding** ([*NamedSharding*](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding") *\|* *P* *\|* *None*) – optional, Specifies how the output array should be sharded across devices in multi-device computation. Can be a [`NamedSharding`](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding"), a [`PartitionSpec`](../jax.sharding.html#jax.sharding.PartitionSpec "jax.sharding.PartitionSpec") (`P`), or `None` (default). When specified, the output will be sharded according to the given sharding specification. Primarily used in explicit sharding mode. See the [explicit sharding tutorial](https://docs.jax.dev/en/latest/parallel.html) for more details.

Returns:  
A random array with the specified dtype and with shape given by `shape` if `shape` is not None, or else by `p.shape`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.random.generalized_normal.html "previous page")

previous

jax.random.generalized_normal

[](jax.random.gumbel.html "next page")

next

jax.random.gumbel

Contents

- [`geometric()`](#jax.random.geometric)

By The JAX authors

© Copyright 2024, The JAX Authors.\
