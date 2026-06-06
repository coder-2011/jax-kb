- [](../index.html)
- [API Reference](../jax.html)
- [`jax.random` module](../jax.random.html)
- jax.random.beta

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.random.beta.rst "Download source file")
-  .pdf

# jax.random.beta

## Contents

- [`beta()`](#jax.random.beta)

# jax.random.beta[\#](#jax-random-beta "Link to this heading")

jax.random.beta(*key*, *a*, *b*, *shape=None*, *dtype=None*, *\**, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/random/core.py#L1224-L1276)[\#](#jax.random.beta "Link to this definition")  
Sample Beta random values with given shape and float dtype.

The values are distributed according to the probability density function:

\\f(x;a,b) \propto x^{a - 1}(1 - x)^{b - 1}\\

on the domain \\0 \le x \le 1\\.

Parameters:  
- **key** (*ArrayLike*) – a PRNG key used as the random key.

- **a** (*RealArray*) – a float or array of floats broadcast-compatible with `shape` representing the first parameter “alpha”.

- **b** (*RealArray*) – a float or array of floats broadcast-compatible with `shape` representing the second parameter “beta”.

- **shape** (*Shape* *\|* *None*) – optional, a tuple of nonnegative integers specifying the result shape. Must be broadcast-compatible with `a` and `b`. The default (None) produces a result shape by broadcasting `a` and `b`.

- **dtype** (*DTypeLikeFloat* *\|* *None*) – optional, a float dtype for the returned values (default float64 if jax_enable_x64 is true, otherwise float32).

- **out_sharding** ([*NamedSharding*](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding") *\|* *P* *\|* *None*) – Optional. Specifies how the output array should be sharded across devices in multi-device computation. Can be a [`NamedSharding`](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding"), a [`PartitionSpec`](../jax.sharding.html#jax.sharding.PartitionSpec "jax.sharding.PartitionSpec") (`P`), or `None` (default). When specified, the output will be sharded according to the given sharding specification. Primarily used in explicit sharding mode. See the [explicit sharding tutorial](https://docs.jax.dev/en/latest/parallel.html) for more details.

Returns:  
A random array with the specified dtype and shape given by `shape` if `shape` is not None, or else by broadcasting `a` and `b`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.random.bernoulli.html "previous page")

previous

jax.random.bernoulli

[](jax.random.binomial.html "next page")

next

jax.random.binomial

Contents

- [`beta()`](#jax.random.beta)

By The JAX authors

© Copyright 2024, The JAX Authors.\
