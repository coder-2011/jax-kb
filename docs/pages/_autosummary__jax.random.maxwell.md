- [](../index.html)
- [API Reference](../jax.html)
- [`jax.random` module](../jax.random.html)
- jax.random.maxwell

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.random.maxwell.rst "Download source file")
-  .pdf

# jax.random.maxwell

## Contents

- [`maxwell()`](#jax.random.maxwell)

# jax.random.maxwell[\#](#jax-random-maxwell "Link to this heading")

jax.random.maxwell(*key*, *shape=()*, *dtype=None*, *\**, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/random/core.py#L2488-L2531)[\#](#jax.random.maxwell "Link to this definition")  
Sample from a one sided Maxwell distribution.

The values are distributed according to the probability density function:

\\f(x) \propto x^2 e^{-x^2 / 2}\\

on the domain \\0 \le x \< \infty\\.

Parameters:  
- **key** (*ArrayLike*) – a PRNG key.

- **shape** (*Shape*) – optional, the batch dimensions of the result. Default ().

- **dtype** (*DTypeLikeFloat* *\|* *None*) – optional, a float dtype for the returned values (default float64 if jax_enable_x64 is true, otherwise float32).

- **out_sharding** ([*NamedSharding*](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding") *\|* *P* *\|* *None*) – optional, specifies how the output array should be sharded across devices in multi-device computation. Can be a [`NamedSharding`](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding"), a [`PartitionSpec`](../jax.sharding.html#jax.sharding.PartitionSpec "jax.sharding.PartitionSpec") (`P`), or `None` (default). When specified, the output will be sharded according to the given sharding specification. Primarily used in explicit sharding mode. See the [explicit sharding tutorial](https://docs.jax.dev/en/latest/parallel.html) for more details.

Returns:  
A jnp.array of samples, of shape shape.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.random.lognormal.html "previous page")

previous

jax.random.lognormal

[](jax.random.multinomial.html "next page")

next

jax.random.multinomial

Contents

- [`maxwell()`](#jax.random.maxwell)

By The JAX authors

© Copyright 2024, The JAX Authors.\
