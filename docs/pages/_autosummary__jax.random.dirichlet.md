- [](../index.html)
- [API Reference](../jax.html)
- [`jax.random` module](../jax.random.html)
- jax.random.dirichlet

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.random.dirichlet.rst "Download source file")
-  .pdf

# jax.random.dirichlet

## Contents

- [`dirichlet()`](#jax.random.dirichlet)

# jax.random.dirichlet[\#](#jax-random-dirichlet "Link to this heading")

jax.random.dirichlet(*key*, *alpha*, *shape=None*, *dtype=None*, *\**, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/random/core.py#L1351-L1407)[\#](#jax.random.dirichlet "Link to this definition")  
Sample Dirichlet random values with given shape and float dtype.

The values are distributed according to the probability density function:

\\f(\\x_i\\; \\\alpha_i\\) \propto \prod\_{i=1}^k x_i^{\alpha_i - 1}\\

Where \\k\\ is the dimension, and \\\\x_i\\\\ satisfies

\\\sum\_{i=1}^k x_i = 1\\

and \\0 \le x_i \le 1\\ for all \\x_i\\.

Parameters:  
- **key** (*ArrayLike*) – a PRNG key used as the random key.

- **alpha** (*RealArray*) – an array of shape `(...,`` ``n)` used as the concentration parameter of the random variables.

- **shape** (*Shape* *\|* *None*) – optional, a tuple of nonnegative integers specifying the result batch shape; that is, the prefix of the result shape excluding the last element of value `n`. Must be broadcast-compatible with `alpha.shape[:-1]`. The default (None) produces a result shape equal to `alpha.shape`.

- **dtype** (*DTypeLikeFloat* *\|* *None*) – optional, a float dtype for the returned values (default float64 if jax_enable_x64 is true, otherwise float32).

- **out_sharding** ([*NamedSharding*](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding") *\|* *P* *\|* *None*) – Optional. Specifies how the output array should be sharded across devices in multi-device computation. Can be a [`NamedSharding`](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding"), a [`PartitionSpec`](../jax.sharding.html#jax.sharding.PartitionSpec "jax.sharding.PartitionSpec") (`P`), or `None` (default). When specified, the output will be sharded according to the given sharding specification. Primarily used in explicit sharding mode. See the [explicit sharding tutorial](https://docs.jax.dev/en/latest/parallel.html) for more details.

Returns:  
A random array with the specified dtype and shape given by `shape`` ``+`` ``(alpha.shape[-1],)` if `shape` is not None, or else `alpha.shape`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.random.choice.html "previous page")

previous

jax.random.choice

[](jax.random.double_sided_maxwell.html "next page")

next

jax.random.double_sided_maxwell

Contents

- [`dirichlet()`](#jax.random.dirichlet)

By The JAX authors

© Copyright 2024, The JAX Authors.\
