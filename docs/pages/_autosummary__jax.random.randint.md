- [](../index.html)
- [API Reference](../jax.html)
- [`jax.random` module](../jax.random.html)
- jax.random.randint

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.random.randint.rst "Download source file")
-  .pdf

# jax.random.randint

## Contents

- [`randint()`](#jax.random.randint)

# jax.random.randint[\#](#jax-random-randint "Link to this heading")

jax.random.randint(*key*, *shape*, *minval*, *maxval*, *dtype=None*, *\**, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/random/core.py#L593-L663)[\#](#jax.random.randint "Link to this definition")  
Sample uniform random values in \[minval, maxval) with given shape/dtype.

Parameters:  
- **key** (*ArrayLike*) – a PRNG key used as the random key.

- **shape** (*Shape*) – a tuple of nonnegative integers representing the shape.

- **minval** (*IntegerArray*) – int or array of ints broadcast-compatible with `shape`, a minimum (inclusive) value for the range.

- **maxval** (*IntegerArray*) – int or array of ints broadcast-compatible with `shape`, a maximum (exclusive) value for the range.

- **dtype** (*DTypeLikeInt* *\|* *None*) – optional, an int dtype for the returned values (default int64 if jax_enable_x64 is true, otherwise int32).

- **out_sharding** ([*NamedSharding*](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding") *\|* *P* *\|* *None*) – Optional. Specifies how the output array should be sharded across devices in multi-device computation. Can be a [`NamedSharding`](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding"), a [`PartitionSpec`](../jax.sharding.html#jax.sharding.PartitionSpec "jax.sharding.PartitionSpec") (`P`), or `None` (default). When specified, the output will be sharded according to the given sharding specification. Primarily used in explicit sharding mode. See the [explicit sharding tutorial](https://docs.jax.dev/en/latest/parallel.html) for more details.

Returns:  
A random array with the specified shape and dtype.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Note

[`randint()`](#jax.random.randint "jax.random.randint") uses a modulus-based computation that is known to produce slightly biased values in some cases. The magnitude of the bias scales as `(maxval`` ``-`` ``minval)`` ``*`` ``((2`` ``**`` ``nbits`` ``)`` ``%`` ``(maxval`` ``-`` ``minval))`` ``/`` ``2`` ``**`` ``nbits`: in words, the bias goes to zero when `(maxval`` ``-`` ``minval)` is a power of 2, and otherwise the bias will be small whenever `(maxval`` ``-`` ``minval)` is small compared to the range of the sampled type.

To reduce this bias, 8-bit and 16-bit values will always be sampled at 32-bit and then cast to the requested type. If you find yourself sampling values for which this bias may be problematic, a possible alternative is to sample via uniform:

    def randint_via_uniform(key, shape, minval, maxval, dtype):
      u = jax.random.uniform(key, shape, minval=minval - 0.5, maxval=maxval - 0.5)
      return u.round().astype(dtype)

But keep in mind this method has its own biases due to floating point rounding errors, and in particular there may be some integers in the range `[minval,`` ``maxval)` that are impossible to produce with this approach.

[](jax.random.rademacher.html "previous page")

previous

jax.random.rademacher

[](jax.random.rayleigh.html "next page")

next

jax.random.rayleigh

Contents

- [`randint()`](#jax.random.randint)

By The JAX authors

© Copyright 2024, The JAX Authors.\
