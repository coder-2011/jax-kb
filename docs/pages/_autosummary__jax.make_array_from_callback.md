- [](../index.html)
- [API Reference](../jax.html)
- jax.make_array_from_callback

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.make_array_from_callback.rst "Download source file")
-  .pdf

# jax.make_array_from_callback

## Contents

- [`make_array_from_callback()`](#jax.make_array_from_callback)

# jax.make_array_from_callback[\#](#jax-make-array-from-callback "Link to this heading")

jax.make_array_from_callback(*shape*, *sharding*, *data_callback*, *dtype=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/array.py#L686-L819)[\#](#jax.make_array_from_callback "Link to this definition")  
Returns a `jax.Array` via data fetched from `data_callback`.

`data_callback` is used to fetch the data for each addressable shard of the returned `jax.Array`. This function must return concrete arrays, meaning that `make_array_from_callback` has limited compatibility with JAX transformations like [`jit()`](jax.jit.html#jax.jit "jax.jit") or [`vmap()`](jax.vmap.html#jax.vmap "jax.vmap").

Parameters:  
- **shape** (*Shape*) – Shape of the `jax.Array`.

- **sharding** ([*Sharding*](../jax.sharding.html#jax.sharding.Sharding "jax.sharding.Sharding") *\|* *Format*) – A `Sharding` instance which describes how the `jax.Array` is laid out across devices.

- **data_callback** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[\[Index* *\|* *None\],* *ArrayLike\]*) – Callback that takes indices into the global array value as input and returns the corresponding data of the global array value. The data can be returned as any array-like object, e.g. a `numpy.ndarray`.

- **dtype** (*DTypeLike* *\|* *None*) – The dtype of the output `jax.Array`. If not provided, the dtype of the data for the first addressable shard is used. If there are no addressable shards, the `dtype` argument must be provided.

Returns:  
A `jax.Array` via data fetched from `data_callback`.

Return type:  
ArrayImpl

Examples

    >>> import math
    >>> from jax.sharding import Mesh
    >>> from jax.sharding import PartitionSpec as P
    >>> import numpy as np
    ...
    >>> input_shape = (8, 8)
    >>> global_input_data = np.arange(math.prod(input_shape)).reshape(input_shape)
    >>> global_mesh = Mesh(np.array(jax.devices()).reshape(2, 4), ('x', 'y'))
    >>> inp_sharding = jax.sharding.NamedSharding(global_mesh, P('x', 'y'))
    ...
    >>> def cb(index):
    ...  return global_input_data[index]
    ...
    >>> arr = jax.make_array_from_callback(input_shape, inp_sharding, cb)
    >>> arr.addressable_data(0).shape
    (4, 2)

[](jax.Array.html "previous page")

previous

jax.Array

[](jax.make_array_from_single_device_arrays.html "next page")

next

jax.make_array_from_single_device_arrays

Contents

- [`make_array_from_callback()`](#jax.make_array_from_callback)

By The JAX authors

© Copyright 2024, The JAX Authors.\
