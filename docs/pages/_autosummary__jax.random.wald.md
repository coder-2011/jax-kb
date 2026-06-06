- [](../index.html)
- [API Reference](../jax.html)
- [`jax.random` module](../jax.random.html)
- jax.random.wald

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.random.wald.rst "Download source file")
-  .pdf

# jax.random.wald

## Contents

- [`wald()`](#jax.random.wald)

# jax.random.wald[\#](#jax-random-wald "Link to this heading")

jax.random.wald(*key*, *mean*, *shape=None*, *dtype=None*, *\**, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/random/core.py#L2872-L2921)[\#](#jax.random.wald "Link to this definition")  
Sample Wald random values with given shape and float dtype.

The values are returned according to the probability density function:

\\f(x;\mu) = \frac{1}{\sqrt{2\pi x^3}} \exp\left(-\frac{(x - \mu)^2}{2\mu^2 x}\right)\\

on the domain \\-\infty \< x \< \infty\\, and where \\\mu \> 0\\ is the location parameter of the distribution.

Parameters:  
- **key** (*ArrayLike*) – a PRNG key used as the random key.

- **mean** (*RealArray*) – a float or array of floats broadcast-compatible with `shape` representing the mean parameter of the distribution.

- **shape** (*Shape* *\|* *None*) – optional, a tuple of nonnegative integers specifying the result shape. Must be broadcast-compatible with `mean`. The default (None) produces a result shape equal to `np.shape(mean)`.

- **dtype** (*DTypeLikeFloat* *\|* *None*) – optional, a float dtype for the returned values (default float64 if jax_enable_x64 is true, otherwise float32).

- **out_sharding** ([*NamedSharding*](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding") *\|* *P* *\|* *None*) – optional, specifies how the output array should be sharded across devices in multi-device computation. Can be a [`NamedSharding`](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding"), a [`PartitionSpec`](../jax.sharding.html#jax.sharding.PartitionSpec "jax.sharding.PartitionSpec") (`P`), or `None` (default). When specified, the output will be sharded according to the given sharding specification. Primarily used in explicit sharding mode. See the [explicit sharding tutorial](https://docs.jax.dev/en/latest/parallel.html) for more details.

Returns:  
A random array with the specified dtype and with shape given by `shape` if `shape` is not None, or else by `mean.shape`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.random.uniform.html "previous page")

previous

jax.random.uniform

[](jax.random.weibull_min.html "next page")

next

jax.random.weibull_min

Contents

- [`wald()`](#jax.random.wald)

By The JAX authors

© Copyright 2024, The JAX Authors.\
