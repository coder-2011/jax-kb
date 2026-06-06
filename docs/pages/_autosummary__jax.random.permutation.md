- [](../index.html)
- [API Reference](../jax.html)
- [`jax.random` module](../jax.random.html)
- jax.random.permutation

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.random.permutation.rst "Download source file")
-  .pdf

# jax.random.permutation

## Contents

- [`permutation()`](#jax.random.permutation)

# jax.random.permutation[\#](#jax-random-permutation "Link to this heading")

jax.random.permutation(*key*, *x*, *axis=0*, *independent=False*, *\**, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/random/core.py#L745-L784)[\#](#jax.random.permutation "Link to this definition")  
Returns a randomly permuted array or range.

Parameters:  
- **key** (*ArrayLike*) – a PRNG key used as the random key.

- **x** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *ArrayLike*) – int or array. If x is an integer, randomly shuffle np.arange(x). If x is an array, randomly shuffle its elements.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – int, optional. The axis which x is shuffled along. Default is 0.

- **independent** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – bool, optional. If set to True, each individual vector along the given axis is shuffled independently. Default is False.

- **out_sharding** ([*NamedSharding*](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding") *\|* *P* *\|* *None*) – Optional. Specifies how the output array should be sharded across devices in multi-device computation. Can be a [`NamedSharding`](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding"), a [`PartitionSpec`](../jax.sharding.html#jax.sharding.PartitionSpec "jax.sharding.PartitionSpec") (`P`), or `None` (default). When specified, the output will be sharded according to the given sharding specification. Primarily used in explicit sharding mode. See the [explicit sharding tutorial](https://docs.jax.dev/en/latest/parallel.html) for more details.

Returns:  
A shuffled version of x or array range

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.random.pareto.html "previous page")

previous

jax.random.pareto

[](jax.random.poisson.html "next page")

next

jax.random.poisson

Contents

- [`permutation()`](#jax.random.permutation)

By The JAX authors

© Copyright 2024, The JAX Authors.\
