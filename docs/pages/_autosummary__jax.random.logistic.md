- [](../index.html)
- [API Reference](../jax.html)
- [`jax.random` module](../jax.random.html)
- jax.random.logistic

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.random.logistic.rst "Download source file")
-  .pdf

# jax.random.logistic

## Contents

- [`logistic()`](#jax.random.logistic)

# jax.random.logistic[\#](#jax-random-logistic "Link to this heading")

jax.random.logistic(*key*, *shape=()*, *dtype=None*, *\**, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/random/core.py#L2145-L2185)[\#](#jax.random.logistic "Link to this definition")  
Sample logistic random values with given shape and float dtype.

The values are distributed according to the probability density function:

\\f(x) = \frac{e^{-x}}{(1 + e^{-x})^2}\\

Parameters:  
- **key** (*ArrayLike*) – a PRNG key used as the random key.

- **shape** (*Shape*) – optional, a tuple of nonnegative integers representing the result shape. Default ().

- **dtype** (*DTypeLikeFloat* *\|* *None*) – optional, a float dtype for the returned values (default float64 if jax_enable_x64 is true, otherwise float32).

- **out_sharding** ([*NamedSharding*](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding") *\|* *P* *\|* *None*) – Optional. Specifies how the output array should be sharded across devices in multi-device computation. Can be a [`NamedSharding`](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding"), a [`PartitionSpec`](../jax.sharding.html#jax.sharding.PartitionSpec "jax.sharding.PartitionSpec") (`P`), or `None` (default). When specified, the output will be sharded according to the given sharding specification. Primarily used in explicit sharding mode. See the [explicit sharding tutorial](https://docs.jax.dev/en/latest/parallel.html) for more details.

Returns:  
A random array with the specified shape and dtype.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.random.loggamma.html "previous page")

previous

jax.random.loggamma

[](jax.random.lognormal.html "next page")

next

jax.random.lognormal

Contents

- [`logistic()`](#jax.random.logistic)

By The JAX authors

© Copyright 2024, The JAX Authors.\
