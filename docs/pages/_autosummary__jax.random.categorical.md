- [](../index.html)
- [API Reference](../jax.html)
- [`jax.random` module](../jax.random.html)
- jax.random.categorical

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.random.categorical.rst "Download source file")
-  .pdf

# jax.random.categorical

## Contents

- [`categorical()`](#jax.random.categorical)

# jax.random.categorical[\#](#jax-random-categorical "Link to this heading")

jax.random.categorical(*key*, *logits*, *axis=-1*, *shape=None*, *replace=True*, *mode=None*, *\**, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/random/core.py#L2001-L2064)[\#](#jax.random.categorical "Link to this definition")  
Sample random values from categorical distributions.

Sampling with replacement uses the Gumbel max trick. Sampling without replacement uses the Gumbel top-k trick. See \[1\] for reference.

Parameters:  
- **key** (*ArrayLike*) – a PRNG key used as the random key.

- **logits** (*RealArray*) – Unnormalized log probabilities of the categorical distribution(s) to sample from, so that softmax(logits, axis) gives the corresponding probabilities.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – Axis along which logits belong to the same categorical distribution.

- **shape** (*Shape* *\|* *None*) – Optional, a tuple of nonnegative integers representing the result shape. Must be broadcast-compatible with `np.delete(logits.shape,`` ``axis)`. The default (None) produces a result shape equal to `np.delete(logits.shape,`` ``axis)`.

- **replace** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – If True (default), perform sampling with replacement. If False, perform sampling without replacement.

- **mode** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – optional, “high” or “low” for how many bits to use in the gumbel sampler. The default is determined by the `use_high_dynamic_range_gumbel` config, which defaults to “low”. With mode=”low”, in float32 sampling will be biased for events with probability less than about 1E-7; with mode=”high” this limit is pushed down to about 1E-14. mode=”high” approximately doubles the cost of sampling.

- **out_sharding** ([*NamedSharding*](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding") *\|* *P* *\|* *None*) – Optional. Specifies how the output array should be sharded across devices in multi-device computation. Can be a [`NamedSharding`](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding"), a [`PartitionSpec`](../jax.sharding.html#jax.sharding.PartitionSpec "jax.sharding.PartitionSpec") (`P`), or `None` (default). When specified, the output will be sharded according to the given sharding specification. Primarily used in explicit sharding mode. See the [explicit sharding tutorial](https://docs.jax.dev/en/latest/parallel.html) for more details.

Returns:  
A random array with int dtype and shape given by `shape` if `shape` is not None, or else `np.delete(logits.shape,`` ``axis)`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

References

\[1\]

Wouter Kool, Herke van Hoof, Max Welling. “Stochastic Beams and Where to Find Them: The Gumbel-Top-k Trick for Sampling Sequences Without Replacement”. Proceedings of the 36th International Conference on Machine Learning, PMLR 97:3499-3508, 2019. [https://proceedings.mlr.press/v97/kool19a.html](https://proceedings.mlr.press/v97/kool19a.html).

[](jax.random.bits.html "previous page")

previous

jax.random.bits

[](jax.random.cauchy.html "next page")

next

jax.random.cauchy

Contents

- [`categorical()`](#jax.random.categorical)

By The JAX authors

© Copyright 2024, The JAX Authors.\
