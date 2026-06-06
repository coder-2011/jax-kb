- [](../index.html)
- [API Reference](../jax.html)
- [`jax.random` module](../jax.random.html)
- jax.random.f

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.random.f.rst "Download source file")
-  .pdf

# jax.random.f

## Contents

- [`f()`](#jax.random.f)

# jax.random.f[\#](#jax-random-f "Link to this heading")

jax.random.f(*key*, *dfnum*, *dfden*, *shape=None*, *dtype=None*, *\**, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/random/core.py#L2373-L2427)[\#](#jax.random.f "Link to this definition")  
Sample F-distribution random values with given shape and float dtype.

The values are distributed according to the probability density function:

\\f(x; \nu_1, \nu_2) \propto x^{\nu_1/2 - 1}\left(1 + \frac{\nu_1}{\nu_2}x\right)^{ -(\nu_1 + \nu_2) / 2}\\

on the domain \\0 \< x \< \infty\\. Here \\\nu_1\\ is the degrees of freedom of the numerator (`dfnum`), and \\\nu_2\\ is the degrees of freedom of the denominator (`dfden`).

Parameters:  
- **key** (*ArrayLike*) – a PRNG key used as the random key.

- **dfnum** (*RealArray*) – a float or array of floats broadcast-compatible with `shape` representing the numerator’s `df` of the distribution.

- **dfden** (*RealArray*) – a float or array of floats broadcast-compatible with `shape` representing the denominator’s `df` of the distribution.

- **shape** (*Shape* *\|* *None*) – optional, a tuple of nonnegative integers specifying the result shape. Must be broadcast-compatible with `dfnum` and `dfden`. The default (None) produces a result shape equal to `dfnum.shape`, and `dfden.shape`.

- **dtype** (*DTypeLikeFloat* *\|* *None*) – optional, a float dtype for the returned values (default float64 if jax_enable_x64 is true, otherwise float32).

- **out_sharding** ([*NamedSharding*](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding") *\|* *P* *\|* *None*) – optional, specifies how the output array should be sharded across devices in multi-device computation. Can be a [`NamedSharding`](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding"), a [`PartitionSpec`](../jax.sharding.html#jax.sharding.PartitionSpec "jax.sharding.PartitionSpec") (`P`), or `None` (default). When specified, the output will be sharded according to the given sharding specification. Primarily used in explicit sharding mode. See the [explicit sharding tutorial](https://docs.jax.dev/en/latest/parallel.html) for more details.

Returns:  
A random array with the specified dtype and with shape given by `shape` if `shape` is not None, or else by `df.shape`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.random.exponential.html "previous page")

previous

jax.random.exponential

[](jax.random.gamma.html "next page")

next

jax.random.gamma

Contents

- [`f()`](#jax.random.f)

By The JAX authors

© Copyright 2024, The JAX Authors.\
