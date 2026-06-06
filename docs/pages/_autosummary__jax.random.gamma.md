- [](../index.html)
- [API Reference](../jax.html)
- [`jax.random` module](../jax.random.html)
- jax.random.gamma

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.random.gamma.rst "Download source file")
-  .pdf

# jax.random.gamma

## Contents

- [`gamma()`](#jax.random.gamma)

# jax.random.gamma[\#](#jax-random-gamma "Link to this heading")

jax.random.gamma(*key*, *a*, *shape=None*, *dtype=None*, *\**, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/random/core.py#L1624-L1680)[\#](#jax.random.gamma "Link to this definition")  
Sample Gamma random values with given shape and float dtype.

The values are distributed according to the probability density function:

\\f(x;a) \propto x^{a - 1} e^{-x}\\

on the domain \\0 \le x \< \infty\\, with \\a \> 0\\.

This is the standard gamma density, with a unit scale/rate parameter. Dividing the sample output by the rate is equivalent to sampling from *gamma(a, rate)*, and multiplying the sample output by the scale is equivalent to sampling from *gamma(a, scale)*.

Parameters:  
- **key** (*ArrayLike*) – a PRNG key used as the random key.

- **a** (*RealArray*) – a float or array of floats broadcast-compatible with `shape` representing the parameter of the distribution.

- **shape** (*Shape* *\|* *None*) – optional, a tuple of nonnegative integers specifying the result shape. Must be broadcast-compatible with `a`. The default (None) produces a result shape equal to `a.shape`.

- **dtype** (*DTypeLikeFloat* *\|* *None*) – optional, a float dtype for the returned values (default float64 if jax_enable_x64 is true, otherwise float32).

- **out_sharding** ([*NamedSharding*](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding") *\|* *P* *\|* *None*) – Optional. Specifies how the output array should be sharded across devices in multi-device computation. Can be a [`NamedSharding`](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding"), a [`PartitionSpec`](../jax.sharding.html#jax.sharding.PartitionSpec "jax.sharding.PartitionSpec") (`P`), or `None` (default). When specified, the output will be sharded according to the given sharding specification. Primarily used in explicit sharding mode. See the [explicit sharding tutorial](https://docs.jax.dev/en/latest/parallel.html) for more details.

Returns:  
A random array with the specified dtype and with shape given by `shape` if `shape` is not None, or else by `a.shape`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

loggammasample gamma values in log-space, which can provide improved  
accuracy for small values of `a`.

[](jax.random.f.html "previous page")

previous

jax.random.f

[](jax.random.generalized_normal.html "next page")

next

jax.random.generalized_normal

Contents

- [`gamma()`](#jax.random.gamma)

By The JAX authors

© Copyright 2024, The JAX Authors.\
