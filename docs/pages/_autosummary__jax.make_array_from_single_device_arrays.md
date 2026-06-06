- [](../index.html)
- [API Reference](../jax.html)
- jax.make_array_from_single_device_arrays

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.make_array_from_single_device_arrays.rst "Download source file")
-  .pdf

# jax.make_array_from_single_device_arrays

## Contents

- [`make_array_from_single_device_arrays()`](#jax.make_array_from_single_device_arrays)

# jax.make_array_from_single_device_arrays[\#](#jax-make-array-from-single-device-arrays "Link to this heading")

jax.make_array_from_single_device_arrays(*shape*, *sharding*, *arrays*, *\**, *dtype=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/array.py#L1022-L1095)[\#](#jax.make_array_from_single_device_arrays "Link to this definition")  
Returns a `jax.Array` from a sequence of `jax.Array`s each on a single device.  
Every device in input `sharding`'s mesh must have an array in `arrays`s.

Parameters:  
- **shape** (*Shape*) – Shape of the output `jax.Array`. This conveys information already included with `sharding` and `arrays` and serves as a double check.

- **sharding** ([*Sharding*](../jax.sharding.html#jax.sharding.Sharding "jax.sharding.Sharding")) – Sharding: A global Sharding instance which describes how the output jax.Array is laid out across devices.

- **arrays** (*Sequence\[basearray.Array\]*) – list or tuple of `jax.Array`s that are each single device addressable. `len(arrays)` must equal `len(sharding.addressable_devices)` and the shape of each array must be the same. For multiprocess code, each process will call with a different `arrays` argument that corresponds to that processes’ data. These arrays are commonly created via `jax.device_put`.

- **dtype** (*DTypeLike* *\|* *None*) – The dtype of the output `jax.Array`. If not provided, the dtype of the first array in `arrays` is used. If `arrays` is empty, the `dtype` argument must be provided.

Returns:  
A global `jax.Array`, sharded as `sharding`, with shape equal to `shape`, and with per-device  
contents matching `arrays`.

Return type:  
ArrayImpl

Examples

    >>> import math
    >>> from jax.sharding import Mesh
    >>> from jax.sharding import PartitionSpec as P
    >>> import numpy as np
    ...
    >>> mesh_rows = 2
    >>> mesh_cols =  jax.device_count() // 2
    ...
    >>> global_shape = (8, 8)
    >>> mesh = Mesh(np.array(jax.devices()).reshape(mesh_rows, mesh_cols), ('x', 'y'))
    >>> sharding = jax.sharding.NamedSharding(mesh, P('x', 'y'))
    >>> inp_data = np.arange(math.prod(global_shape)).reshape(global_shape)
    ...
    >>> arrays = [
    ...    jax.device_put(inp_data[index], d)
    ...        for d, index in sharding.addressable_devices_indices_map(global_shape).items()]
    ...
    >>> arr = jax.make_array_from_single_device_arrays(global_shape, sharding, arrays)
    >>> assert arr.shape == (8,8) # arr.shape is (8,8) regardless of jax.device_count()

For cases where you have a local array and want to convert it to a global jax.Array, use `jax.make_array_from_process_local_data`.

[](jax.make_array_from_callback.html "previous page")

previous

jax.make_array_from_callback

[](jax.make_array_from_process_local_data.html "next page")

next

jax.make_array_from_process_local_data

Contents

- [`make_array_from_single_device_arrays()`](#jax.make_array_from_single_device_arrays)

By The JAX authors

© Copyright 2024, The JAX Authors.\
