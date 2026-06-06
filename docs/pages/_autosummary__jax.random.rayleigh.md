- [](../index.html)
- [API Reference](../jax.html)
- [`jax.random` module](../jax.random.html)
- jax.random.rayleigh

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.random.rayleigh.rst "Download source file")
-  .pdf

# jax.random.rayleigh

## Contents

- [`rayleigh()`](#jax.random.rayleigh)

# jax.random.rayleigh[\#](#jax-random-rayleigh "Link to this heading")

jax.random.rayleigh(*key*, *scale*, *shape=None*, *dtype=None*, *\**, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/random/core.py#L2811-L2860)[\#](#jax.random.rayleigh "Link to this definition")  
Sample Rayleigh random values with given shape and float dtype.

The values are returned according to the probability density function:

\\f(x;\sigma) \propto xe^{-x^2/(2\sigma^2)}\\

on the domain \\-\infty \< x \< \infty\\, and where \\\sigma \> 0\\ is the scale parameter of the distribution.

Parameters:  
- **key** (*ArrayLike*) – a PRNG key used as the random key.

- **scale** (*RealArray*) – a float or array of floats broadcast-compatible with `shape` representing the parameter of the distribution.

- **shape** (*Shape* *\|* *None*) – optional, a tuple of nonnegative integers specifying the result shape. Must be broadcast-compatible with `scale`. The default (None) produces a result shape equal to `scale.shape`.

- **dtype** (*DTypeLikeFloat* *\|* *None*) – optional, a float dtype for the returned values (default float64 if jax_enable_x64 is true, otherwise float32).

- **out_sharding** ([*NamedSharding*](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding") *\|* *P* *\|* *None*) – Optional. Specifies how the output array should be sharded across devices in multi-device computation. Can be a [`NamedSharding`](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding"), a [`PartitionSpec`](../jax.sharding.html#jax.sharding.PartitionSpec "jax.sharding.PartitionSpec") (`P`), or `None` (default). When specified, the output will be sharded according to the given sharding specification. Primarily used in explicit sharding mode. See the [explicit sharding tutorial](https://docs.jax.dev/en/latest/parallel.html) for more details.

Returns:  
A random array with the specified dtype and with shape given by `shape` if `shape` is not None, or else by `scale.shape`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.random.randint.html "previous page")

previous

jax.random.randint

[](jax.random.t.html "next page")

next

jax.random.t

Contents

- [`rayleigh()`](#jax.random.rayleigh)

By The JAX authors

© Copyright 2024, The JAX Authors.\
