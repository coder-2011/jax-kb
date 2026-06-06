- [](../index.html)
- [API Reference](../jax.html)
- [`jax.random` module](../jax.random.html)
- jax.random.generalized_normal

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.random.generalized_normal.rst "Download source file")
-  .pdf

# jax.random.generalized_normal

## Contents

- [`generalized_normal()`](#jax.random.generalized_normal)

# jax.random.generalized_normal[\#](#jax-random-generalized-normal "Link to this heading")

jax.random.generalized_normal(*key*, *p*, *shape=()*, *dtype=None*, *\**, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/random/core.py#L2710-L2754)[\#](#jax.random.generalized_normal "Link to this definition")  
Sample from the generalized normal distribution.

The values are returned according to the probability density function:

\\f(x;p) \propto e^{-\|x\|^p}\\

on the domain \\-\infty \< x \< \infty\\, where \\p \> 0\\ is the shape parameter.

Parameters:  
- **key** (*ArrayLike*) – a PRNG key used as the random key.

- **p** ([*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)")) – a float representing the shape parameter.

- **shape** (*Shape*) – optional, the batch dimensions of the result. Default ().

- **dtype** (*DTypeLikeFloat* *\|* *None*) – optional, a float dtype for the returned values (default float64 if jax_enable_x64 is true, otherwise float32).

- **out_sharding** ([*NamedSharding*](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding") *\|* *P* *\|* *None*) – optional, specifies how the output array should be sharded across devices in multi-device computation. Can be a [`NamedSharding`](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding"), a [`PartitionSpec`](../jax.sharding.html#jax.sharding.PartitionSpec "jax.sharding.PartitionSpec") (`P`), or `None` (default). When specified, the output will be sharded according to the given sharding specification. Primarily used in explicit sharding mode. See the [explicit sharding tutorial](https://docs.jax.dev/en/latest/parallel.html) for more details.

Returns:  
A random array with the specified shape and dtype.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.random.gamma.html "previous page")

previous

jax.random.gamma

[](jax.random.geometric.html "next page")

next

jax.random.geometric

Contents

- [`generalized_normal()`](#jax.random.generalized_normal)

By The JAX authors

© Copyright 2024, The JAX Authors.\
