- [](../index.html)
- [API Reference](../jax.html)
- [`jax.random` module](../jax.random.html)
- jax.random.loggamma

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.random.loggamma.rst "Download source file")
-  .pdf

# jax.random.loggamma

## Contents

- [`loggamma()`](#jax.random.loggamma)

# jax.random.loggamma[\#](#jax-random-loggamma "Link to this heading")

jax.random.loggamma(*key*, *a*, *shape=None*, *dtype=None*, *\**, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/random/core.py#L1682-L1733)[\#](#jax.random.loggamma "Link to this definition")  
Sample log-gamma random values with given shape and float dtype.

This function is implemented such that the following will hold for a dtype-appropriate tolerance:

    np.testing.assert_allclose(jnp.exp(loggamma(*args)), gamma(*args), rtol=rtol)

The benefit of log-gamma is that for samples very close to zero (which occur frequently when a \<\< 1) sampling in log space provides better precision.

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

gamma : standard gamma sampler.

[](jax.random.laplace.html "previous page")

previous

jax.random.laplace

[](jax.random.logistic.html "next page")

next

jax.random.logistic

Contents

- [`loggamma()`](#jax.random.loggamma)

By The JAX authors

© Copyright 2024, The JAX Authors.\
