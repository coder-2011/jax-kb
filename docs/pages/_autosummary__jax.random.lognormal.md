- [](../index.html)
- [API Reference](../jax.html)
- [`jax.random` module](../jax.random.html)
- jax.random.lognormal

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.random.lognormal.rst "Download source file")
-  .pdf

# jax.random.lognormal

## Contents

- [`lognormal()`](#jax.random.lognormal)

# jax.random.lognormal[\#](#jax-random-lognormal "Link to this heading")

jax.random.lognormal(*key*, *sigma=np.float32(1.0)*, *shape=None*, *dtype=None*, *\**, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/random/core.py#L3071-L3115)[\#](#jax.random.lognormal "Link to this definition")  
Sample lognormal random values with given shape and float dtype.

The values are distributed according to the probability density function:

\\f(x) = \frac{1}{x\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(\log x)^2}{2\sigma^2}\right)\\

on the domain \\x \> 0\\.

Parameters:  
- **key** (*ArrayLike*) – a PRNG key used as the random key.

- **sigma** (*RealArray*) – a float or array of floats broadcast-compatible with `shape` representing the standard deviation of the underlying normal distribution. Default 1.

- **shape** (*Shape* *\|* *None*) – optional, a tuple of nonnegative integers specifying the result shape. The default (None) produces a result shape equal to `()`.

- **dtype** (*DTypeLikeFloat* *\|* *None*) – optional, a float dtype for the returned values (default float64 if jax_enable_x64 is true, otherwise float32).

- **out_sharding** ([*NamedSharding*](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding") *\|* *P* *\|* *None*) – optional, specifies how the output array should be sharded across devices in multi-device computation. Can be a [`NamedSharding`](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding"), a [`PartitionSpec`](../jax.sharding.html#jax.sharding.PartitionSpec "jax.sharding.PartitionSpec") (`P`), or `None` (default). When specified, the output will be sharded according to the given sharding specification. Primarily used in explicit sharding mode. See the [explicit sharding tutorial](https://docs.jax.dev/en/latest/parallel.html) for more details.

Returns:  
A random array with the specified dtype and with shape given by `shape`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.random.logistic.html "previous page")

previous

jax.random.logistic

[](jax.random.maxwell.html "next page")

next

jax.random.maxwell

Contents

- [`lognormal()`](#jax.random.lognormal)

By The JAX authors

© Copyright 2024, The JAX Authors.\
