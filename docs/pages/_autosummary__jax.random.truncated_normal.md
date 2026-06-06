- [](../index.html)
- [API Reference](../jax.html)
- [`jax.random` module](../jax.random.html)
- jax.random.truncated_normal

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.random.truncated_normal.rst "Download source file")
-  .pdf

# jax.random.truncated_normal

## Contents

- [`truncated_normal()`](#jax.random.truncated_normal)

# jax.random.truncated_normal[\#](#jax-random-truncated-normal "Link to this heading")

jax.random.truncated_normal(*key*, *lower*, *upper*, *shape=None*, *dtype=None*, *\**, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/random/core.py#L1073-L1126)[\#](#jax.random.truncated_normal "Link to this definition")  
Sample truncated standard normal random values with given shape and dtype.

The values are returned according to the probability density function:

\\f(x) \propto e^{-x^2/2}\\

on the domain \\\rm{lower} \< x \< \rm{upper}\\.

Parameters:  
- **key** (*ArrayLike*) – a PRNG key used as the random key.

- **lower** (*RealArray*) – a float or array of floats representing the lower bound for truncation. Must be broadcast-compatible with `upper`.

- **upper** (*RealArray*) – a float or array of floats representing the upper bound for truncation. Must be broadcast-compatible with `lower`.

- **shape** (*Shape* *\|* *None*) – optional, a tuple of nonnegative integers specifying the result shape. Must be broadcast-compatible with `lower` and `upper`. The default (None) produces a result shape by broadcasting `lower` and `upper`.

- **dtype** (*DTypeLikeFloat* *\|* *None*) – optional, a float dtype for the returned values (default float64 if jax_enable_x64 is true, otherwise float32).

- **out_sharding** ([*NamedSharding*](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding") *\|* *P* *\|* *None*) – Optional. Specifies how the output array should be sharded across devices in multi-device computation. Can be a [`NamedSharding`](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding"), a [`PartitionSpec`](../jax.sharding.html#jax.sharding.PartitionSpec "jax.sharding.PartitionSpec") (`P`), or `None` (default). When specified, the output will be sharded according to the given sharding specification. Primarily used in explicit sharding mode. See the [explicit sharding tutorial](https://docs.jax.dev/en/latest/parallel.html) for more details.

Returns:  
A random array with the specified dtype and shape given by `shape` if `shape` is not None, or else by broadcasting `lower` and `upper`. Returns values in the open interval `(lower,`` ``upper)`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.random.triangular.html "previous page")

previous

jax.random.triangular

[](jax.random.uniform.html "next page")

next

jax.random.uniform

Contents

- [`truncated_normal()`](#jax.random.truncated_normal)

By The JAX authors

© Copyright 2024, The JAX Authors.\
