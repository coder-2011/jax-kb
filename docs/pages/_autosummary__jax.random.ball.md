- [](../index.html)
- [API Reference](../jax.html)
- [`jax.random` module](../jax.random.html)
- jax.random.ball

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.random.ball.rst "Download source file")
-  .pdf

# jax.random.ball

## Contents

- [`ball()`](#jax.random.ball)

# jax.random.ball[\#](#jax-random-ball "Link to this heading")

jax.random.ball(*key*, *d*, *p=2*, *shape=()*, *dtype=None*, *\**, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/random/core.py#L2762-L2802)[\#](#jax.random.ball "Link to this definition")  
Sample uniformly from the unit Lp ball.

Reference: [https://arxiv.org/abs/math/0503650](https://arxiv.org/abs/math/0503650).

Parameters:  
- **key** (*ArrayLike*) – a PRNG key used as the random key.

- **d** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – a nonnegative int representing the dimensionality of the ball.

- **p** ([*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)")) – a float representing the p parameter of the Lp norm.

- **shape** (*Shape*) – optional, the batch dimensions of the result. Default ().

- **dtype** (*DTypeLikeFloat* *\|* *None*) – optional, a float dtype for the returned values (default float64 if jax_enable_x64 is true, otherwise float32).

- **out_sharding** ([*NamedSharding*](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding") *\|* *P* *\|* *None*) – optional, specifies how the output array should be sharded across devices in multi-device computation. Can be a [`NamedSharding`](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding"), a [`PartitionSpec`](../jax.sharding.html#jax.sharding.PartitionSpec "jax.sharding.PartitionSpec") (`P`), or `None` (default). When specified, the output will be sharded according to the given sharding specification. Primarily used in explicit sharding mode. See the [explicit sharding tutorial](https://docs.jax.dev/en/latest/parallel.html) for more details.

Returns:  
A random array of shape (\*shape, d) and specified dtype.

[](jax.random.PRNGKey.html "previous page")

previous

jax.random.PRNGKey

[](jax.random.bernoulli.html "next page")

next

jax.random.bernoulli

Contents

- [`ball()`](#jax.random.ball)

By The JAX authors

© Copyright 2024, The JAX Authors.\
