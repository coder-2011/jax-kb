- [](../index.html)
- [API Reference](../jax.html)
- [`jax.random` module](../jax.random.html)
- jax.random.bits

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.random.bits.rst "Download source file")
-  .pdf

# jax.random.bits

## Contents

- [`bits()`](#jax.random.bits)

# jax.random.bits[\#](#jax-random-bits "Link to this heading")

jax.random.bits(*key*, *shape=()*, *dtype=None*, *\**, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/random/core.py#L421-L459)[\#](#jax.random.bits "Link to this definition")  
Sample uniform bits in the form of unsigned integers.

Parameters:  
- **key** (*ArrayLike*) – a PRNG key used as the random key.

- **shape** (*Shape*) – optional, a tuple of nonnegative integers representing the result shape. Default `()`.

- **dtype** (*DTypeLikeUInt* *\|* *None*) – optional, an unsigned integer dtype for the returned values (default `uint64` if `jax_enable_x64` is true, otherwise `uint32`).

- **out_sharding** ([*NamedSharding*](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding") *\|* *P* *\|* *None*) – Optional. Specifies how the output array should be sharded across devices in multi-device computation. Can be a [`NamedSharding`](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding"), a [`PartitionSpec`](../jax.sharding.html#jax.sharding.PartitionSpec "jax.sharding.PartitionSpec") (`P`), or `None` (default). When specified, the output will be sharded according to the given sharding specification. Primarily used in explicit sharding mode. See the [explicit sharding tutorial](https://docs.jax.dev/en/latest/parallel.html) for more details.

Returns:  
A random array with the specified shape and dtype.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.random.binomial.html "previous page")

previous

jax.random.binomial

[](jax.random.categorical.html "next page")

next

jax.random.categorical

Contents

- [`bits()`](#jax.random.bits)

By The JAX authors

© Copyright 2024, The JAX Authors.\
