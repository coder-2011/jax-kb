- [](../index.html)
- [API Reference](../jax.html)
- [`jax.random` module](../jax.random.html)
- jax.random.pareto

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.random.pareto.rst "Download source file")
-  .pdf

# jax.random.pareto

## Contents

- [`pareto()`](#jax.random.pareto)

# jax.random.pareto[\#](#jax-random-pareto "Link to this heading")

jax.random.pareto(*key*, *b*, *shape=None*, *dtype=None*, *\**, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/random/core.py#L2193-L2241)[\#](#jax.random.pareto "Link to this definition")  
Sample Pareto random values with given shape and float dtype.

The values are distributed according to the probability density function:

\\f(x; b) = b / x^{b + 1}\\

on the domain \\1 \le x \< \infty\\ with \\b \> 0\\

Parameters:  
- **key** (*ArrayLike*) – a PRNG key used as the random key.

- **b** (*RealArray*) – a float or array of floats broadcast-compatible with `shape` representing the parameter of the distribution.

- **shape** (*Shape* *\|* *None*) – optional, a tuple of nonnegative integers specifying the result shape. Must be broadcast-compatible with `b`. The default (None) produces a result shape equal to `b.shape`.

- **dtype** (*DTypeLikeFloat* *\|* *None*) – optional, a float dtype for the returned values (default float64 if jax_enable_x64 is true, otherwise float32).

- **out_sharding** ([*NamedSharding*](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding") *\|* *P* *\|* *None*) – Optional. Specifies how the output array should be sharded across devices in multi-device computation. Can be a [`NamedSharding`](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding"), a [`PartitionSpec`](../jax.sharding.html#jax.sharding.PartitionSpec "jax.sharding.PartitionSpec") (`P`), or `None` (default). When specified, the output will be sharded according to the given sharding specification. Primarily used in explicit sharding mode. See the [explicit sharding tutorial](https://docs.jax.dev/en/latest/parallel.html) for more details.

Returns:  
A random array with the specified dtype and with shape given by `shape` if `shape` is not None, or else by `b.shape`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.random.orthogonal.html "previous page")

previous

jax.random.orthogonal

[](jax.random.permutation.html "next page")

next

jax.random.permutation

Contents

- [`pareto()`](#jax.random.pareto)

By The JAX authors

© Copyright 2024, The JAX Authors.\
