- [](../index.html)
- [API Reference](../jax.html)
- [`jax.ops` module](../jax.ops.html)
- jax.ops.segment_max

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.ops.segment_max.rst "Download source file")
-  .pdf

# jax.ops.segment_max

## Contents

- [`segment_max()`](#jax.ops.segment_max)

# jax.ops.segment_max[\#](#jax-ops-segment-max "Link to this heading")

jax.ops.segment_max(*data*, *segment_ids*, *num_segments=None*, *indices_are_sorted=False*, *unique_indices=False*, *bucket_size=None*, *mode=None*, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/ops/scatter.py#L339-L396)[\#](#jax.ops.segment_max "Link to this definition")  
Computes the maximum within segments of an array.

Similar to TensorFlow’s [segment_max](https://www.tensorflow.org/api_docs/python/tf/math/segment_max)

Parameters:  
- **data** (*ArrayLike*) – an array with the values to be reduced.

- **segment_ids** (*ArrayLike*) – an array with integer dtype that indicates the segments of data (along its leading axis) to be reduced. Values can be repeated and need not be sorted.

- **num_segments** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – optional, an int with nonnegative value indicating the number of segments. The default is set to be the minimum number of segments that would support all indices in `segment_ids`, calculated as `max(segment_ids)`` ``+`` ``1`. Since num_segments determines the size of the output, a static value must be provided to use `segment_max` in a JIT-compiled function.

- **indices_are_sorted** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – whether `segment_ids` is known to be sorted.

- **unique_indices** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – whether segment_ids is known to be free of duplicates.

- **bucket_size** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – size of bucket to group indices into. `segment_max` is performed on each bucket separately. Default `None` means no bucketing.

- **mode** ([*slicing.GatherScatterMode*](../jax.lax.html#jax.lax.GatherScatterMode "jax._src.lax.slicing.GatherScatterMode") *\|* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – a [`jax.lax.GatherScatterMode`](../jax.lax.html#jax.lax.GatherScatterMode "jax.lax.GatherScatterMode") value describing how out-of-bounds indices should be handled. By default, values outside of the range \[0, num_segments) are dropped and do not contribute to the result.

- **out_sharding** ([*NamedSharding*](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding") *\|* *P* *\|* *None*)

Returns:  
An array with shape `(num_segments,)`` ``+`` ``data.shape[1:]` representing the segment maximums.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Examples

Simple 1D segment max:

    >>> data = jnp.arange(6)
    >>> segment_ids = jnp.array([0, 0, 1, 1, 2, 2])
    >>> segment_max(data, segment_ids)
    Array([1, 3, 5], dtype=int32)

Using JIT requires static num_segments:

    >>> from jax import jit
    >>> jit(segment_max, static_argnums=2)(data, segment_ids, 3)
    Array([1, 3, 5], dtype=int32)

[](../jax.ops.html "previous page")

previous

`jax.ops` module

[](jax.ops.segment_min.html "next page")

next

jax.ops.segment_min

Contents

- [`segment_max()`](#jax.ops.segment_max)

By The JAX authors

© Copyright 2024, The JAX Authors.\
