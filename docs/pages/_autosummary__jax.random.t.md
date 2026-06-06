- [](../index.html)
- [API Reference](../jax.html)
- [`jax.random` module](../jax.random.html)
- jax.random.t

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.random.t.rst "Download source file")
-  .pdf

# jax.random.t

## Contents

- [`t()`](#jax.random.t)

# jax.random.t[\#](#jax-random-t "Link to this heading")

jax.random.t(*key*, *df*, *shape=None*, *dtype=None*, *\**, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/random/core.py#L2249-L2297)[\#](#jax.random.t "Link to this definition")  
Sample Student’s t random values with given shape and float dtype.

The values are distributed according to the probability density function:

\\f(t; \nu) \propto \left(1 + \frac{t^2}{\nu}\right)^{-(\nu + 1)/2}\\

Where \\\nu \> 0\\ is the degrees of freedom, given by the parameter `df`.

Parameters:  
- **key** (*ArrayLike*) – a PRNG key used as the random key.

- **df** (*RealArray*) – a float or array of floats broadcast-compatible with `shape` representing the degrees of freedom parameter of the distribution.

- **shape** (*Shape* *\|* *None*) – optional, a tuple of nonnegative integers specifying the result shape. Must be broadcast-compatible with `df`. The default (None) produces a result shape equal to `df.shape`.

- **dtype** (*DTypeLikeFloat* *\|* *None*) – optional, a float dtype for the returned values (default float64 if jax_enable_x64 is true, otherwise float32).

- **out_sharding** ([*NamedSharding*](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding") *\|* *P* *\|* *None*) – Optional. Specifies how the output array should be sharded across devices in multi-device computation. Can be a [`NamedSharding`](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding"), a [`PartitionSpec`](../jax.sharding.html#jax.sharding.PartitionSpec "jax.sharding.PartitionSpec") (`P`), or `None` (default). When specified, the output will be sharded according to the given sharding specification. Primarily used in explicit sharding mode. See the [explicit sharding tutorial](https://docs.jax.dev/en/latest/parallel.html) for more details.

Returns:  
A random array with the specified dtype and with shape given by `shape` if `shape` is not None, or else by `df.shape`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.random.rayleigh.html "previous page")

previous

jax.random.rayleigh

[](jax.random.triangular.html "next page")

next

jax.random.triangular

Contents

- [`t()`](#jax.random.t)

By The JAX authors

© Copyright 2024, The JAX Authors.\
