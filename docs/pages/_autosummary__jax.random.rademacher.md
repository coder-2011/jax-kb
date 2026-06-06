- [](../index.html)
- [API Reference](../jax.html)
- [`jax.random` module](../jax.random.html)
- jax.random.rademacher

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.random.rademacher.rst "Download source file")
-  .pdf

# jax.random.rademacher

## Contents

- [`rademacher()`](#jax.random.rademacher)

# jax.random.rademacher[\#](#jax-random-rademacher "Link to this heading")

jax.random.rademacher(*key*, *shape=()*, *dtype=None*, *\**, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/random/core.py#L2441-L2479)[\#](#jax.random.rademacher "Link to this definition")  
Sample from a Rademacher distribution.

The values are distributed according to the probability mass function:

\\f(k) = \frac{1}{2}(\delta(k - 1) + \delta(k + 1))\\

on the domain \\k \in \\-1, 1\\\\, where \\\delta(x)\\ is the dirac delta function.

Parameters:  
- **key** (*ArrayLike*) – a PRNG key.

- **shape** (*Shape*) – The shape of the returned samples. Default ().

- **dtype** (*DTypeLikeInt* *\|* *None*) – The type used for samples.

- **out_sharding** ([*NamedSharding*](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding") *\|* *P* *\|* *None*) – optional, specifies how the output array should be sharded across devices in multi-device computation. Can be a [`NamedSharding`](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding"), a [`PartitionSpec`](../jax.sharding.html#jax.sharding.PartitionSpec "jax.sharding.PartitionSpec") (`P`), or `None` (default). When specified, the output will be sharded according to the given sharding specification. Primarily used in explicit sharding mode. See the [explicit sharding tutorial](https://docs.jax.dev/en/latest/parallel.html) for more details.

Returns:  
A jnp.array of samples, of shape shape. Each element in the output has a 50% change of being 1 or -1.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.random.poisson.html "previous page")

previous

jax.random.poisson

[](jax.random.randint.html "next page")

next

jax.random.randint

Contents

- [`rademacher()`](#jax.random.rademacher)

By The JAX authors

© Copyright 2024, The JAX Authors.\
