- [](../index.html)
- [API Reference](../jax.html)
- [`jax.random` module](../jax.random.html)
- jax.random.uniform

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.random.uniform.rst "Download source file")
-  .pdf

# jax.random.uniform

## Contents

- [`uniform()`](#jax.random.uniform)

# jax.random.uniform[\#](#jax-random-uniform "Link to this heading")

jax.random.uniform(*key*, *shape=()*, *dtype=None*, *minval=0.0*, *maxval=1.0*, *\**, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/random/core.py#L470-L510)[\#](#jax.random.uniform "Link to this definition")  
Sample uniform random values in \[minval, maxval) with given shape/dtype.

Parameters:  
- **key** (*ArrayLike*) – a PRNG key used as the random key.

- **shape** (*Shape*) – optional, a tuple of nonnegative integers representing the result shape. Default ().

- **dtype** (*DTypeLikeFloat* *\|* *None*) – optional, a float dtype for the returned values (default float64 if jax_enable_x64 is true, otherwise float32).

- **minval** (*RealArray*) – optional, a minimum (inclusive) value broadcast-compatible with shape for the range (default 0).

- **maxval** (*RealArray*) – optional, a maximum (exclusive) value broadcast-compatible with shape for the range (default 1).

- **out_sharding** ([*NamedSharding*](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding") *\|* *P* *\|* *None*) – Optional. Specifies how the output array should be sharded across devices in multi-device computation. Can be a [`NamedSharding`](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding"), a [`PartitionSpec`](../jax.sharding.html#jax.sharding.PartitionSpec "jax.sharding.PartitionSpec") (`P`), or `None` (default). When specified, the output will be sharded according to the given sharding specification. Primarily used in explicit sharding mode. See the [explicit sharding tutorial](https://docs.jax.dev/en/latest/parallel.html) for more details.

Returns:  
A random array with the specified shape and dtype.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.random.truncated_normal.html "previous page")

previous

jax.random.truncated_normal

[](jax.random.wald.html "next page")

next

jax.random.wald

Contents

- [`uniform()`](#jax.random.uniform)

By The JAX authors

© Copyright 2024, The JAX Authors.\
