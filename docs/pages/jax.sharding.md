- [](index.html)
- [API Reference](jax.html)
- `jax.sharding` module

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](_sources/jax.sharding.rst "Download source file")
-  .pdf

# jax.sharding module

## Contents

- [Classes](#classes)
  - [`Sharding`](#jax.sharding.Sharding)
    - [`Sharding.addressable_devices`](#jax.sharding.Sharding.addressable_devices)
    - [`Sharding.addressable_devices_indices_map()`](#jax.sharding.Sharding.addressable_devices_indices_map)
    - [`Sharding.device_set`](#jax.sharding.Sharding.device_set)
    - [`Sharding.devices_indices_map()`](#jax.sharding.Sharding.devices_indices_map)
    - [`Sharding.is_equivalent_to()`](#jax.sharding.Sharding.is_equivalent_to)
    - [`Sharding.is_fully_addressable`](#jax.sharding.Sharding.is_fully_addressable)
    - [`Sharding.is_fully_replicated`](#jax.sharding.Sharding.is_fully_replicated)
    - [`Sharding.memory_kind`](#jax.sharding.Sharding.memory_kind)
    - [`Sharding.num_devices`](#jax.sharding.Sharding.num_devices)
    - [`Sharding.shard_shape()`](#jax.sharding.Sharding.shard_shape)
    - [`Sharding.with_memory_kind()`](#jax.sharding.Sharding.with_memory_kind)
  - [`SingleDeviceSharding`](#jax.sharding.SingleDeviceSharding)
    - [`SingleDeviceSharding.device_set`](#jax.sharding.SingleDeviceSharding.device_set)
    - [`SingleDeviceSharding.devices_indices_map()`](#jax.sharding.SingleDeviceSharding.devices_indices_map)
    - [`SingleDeviceSharding.is_fully_addressable`](#jax.sharding.SingleDeviceSharding.is_fully_addressable)
    - [`SingleDeviceSharding.is_fully_replicated`](#jax.sharding.SingleDeviceSharding.is_fully_replicated)
    - [`SingleDeviceSharding.memory_kind`](#jax.sharding.SingleDeviceSharding.memory_kind)
    - [`SingleDeviceSharding.num_devices`](#jax.sharding.SingleDeviceSharding.num_devices)
    - [`SingleDeviceSharding.with_memory_kind()`](#jax.sharding.SingleDeviceSharding.with_memory_kind)
  - [`NamedSharding`](#jax.sharding.NamedSharding)
    - [`NamedSharding.addressable_devices`](#jax.sharding.NamedSharding.addressable_devices)
    - [`NamedSharding.device_set`](#jax.sharding.NamedSharding.device_set)
    - [`NamedSharding.is_equivalent_to()`](#jax.sharding.NamedSharding.is_equivalent_to)
    - [`NamedSharding.is_fully_addressable`](#jax.sharding.NamedSharding.is_fully_addressable)
    - [`NamedSharding.is_fully_replicated`](#jax.sharding.NamedSharding.is_fully_replicated)
    - [`NamedSharding.memory_kind`](#jax.sharding.NamedSharding.memory_kind)
    - [`NamedSharding.mesh`](#jax.sharding.NamedSharding.mesh)
    - [`NamedSharding.num_devices`](#jax.sharding.NamedSharding.num_devices)
    - [`NamedSharding.spec`](#jax.sharding.NamedSharding.spec)
    - [`NamedSharding.with_memory_kind()`](#jax.sharding.NamedSharding.with_memory_kind)
  - [`PartitionSpec`](#jax.sharding.PartitionSpec)
  - [`Mesh`](#jax.sharding.Mesh)

# `jax.sharding` module[\#](#module-jax.sharding "Link to this heading")

## Classes[\#](#classes "Link to this heading")

*class* jax.sharding.Sharding(*\*args*, *\*\*kwargs*)[\#](#jax.sharding.Sharding "Link to this definition")  
Describes how a [`jax.Array`](_autosummary/jax.Array.html#jax.Array "jax.Array") is laid out across devices.

*property* addressable_devices*: [set](https://docs.python.org/3/library/stdtypes.html#set "(in Python v3.14)")\[[Device](_autosummary/jax.Device.html#jax.Device "jax.Device")\]*[\#](#jax.sharding.Sharding.addressable_devices "Link to this definition")  
The set of devices in the [`Sharding`](#jax.sharding.Sharding "jax.sharding.Sharding") that are addressable by the current process.

addressable_devices_indices_map(*global_shape*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/sharding.py#L174-L182)[\#](#jax.sharding.Sharding.addressable_devices_indices_map "Link to this definition")  
A mapping from addressable devices to the slice of array data each contains.

`addressable_devices_indices_map` contains that part of `device_indices_map` that applies to the addressable devices.

Parameters:  
**global_shape** (*Shape*)

Return type:  
Mapping\[[Device](_autosummary/jax.Device.html#jax.Device "jax.Device"), Index \| None\]

*property* device_set*: [set](https://docs.python.org/3/library/stdtypes.html#set "(in Python v3.14)")\[[Device](_autosummary/jax.Device.html#jax.Device "jax.Device")\]*[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/sharding.py#L99-L107)[\#](#jax.sharding.Sharding.device_set "Link to this definition")  
The set of devices that this [`Sharding`](#jax.sharding.Sharding "jax.sharding.Sharding") spans.

In multi-controller JAX, the set of devices is global, i.e., includes non-addressable devices from other processes.

devices_indices_map(*global_shape*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/sharding.py#L183-L190)[\#](#jax.sharding.Sharding.devices_indices_map "Link to this definition")  
Returns a mapping from devices to the array slices each contains.

The mapping includes all global devices, i.e., including non-addressable devices from other processes.

Parameters:  
**global_shape** (*Shape*)

Return type:  
Mapping\[[Device](_autosummary/jax.Device.html#jax.Device "jax.Device"), Index\]

is_equivalent_to(*other*, *ndim*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/sharding.py#L209-L216)[\#](#jax.sharding.Sharding.is_equivalent_to "Link to this definition")  
Returns `True` if two shardings are equivalent.

Two shardings are equivalent if they place the same logical array shards on the same devices.

Parameters:  
- **self** ([*Sharding*](#jax.sharding.Sharding "jax.sharding.Sharding"))

- **other** ([*Sharding*](#jax.sharding.Sharding "jax.sharding.Sharding"))

- **ndim** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

Return type:  
[bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")

*property* is_fully_addressable*: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")*[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/sharding.py#L117-L126)[\#](#jax.sharding.Sharding.is_fully_addressable "Link to this definition")  
Is this sharding fully addressable?

A sharding is fully addressable if the current process can address all of the devices named in the [`Sharding`](#jax.sharding.Sharding "jax.sharding.Sharding"). `is_fully_addressable` is equivalent to “is_local” in multi-process JAX.

*property* is_fully_replicated*: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")*[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/sharding.py#L108-L116)[\#](#jax.sharding.Sharding.is_fully_replicated "Link to this definition")  
Is this sharding fully replicated?

A sharding is fully replicated if each device has a complete copy of the entire data.

*property* memory_kind*: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)")*[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/sharding.py#L132-L136)[\#](#jax.sharding.Sharding.memory_kind "Link to this definition")  
Returns the memory kind of the sharding.

*property* num_devices*: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/sharding.py#L127-L131)[\#](#jax.sharding.Sharding.num_devices "Link to this definition")  
Number of devices that the sharding contains.

shard_shape(*global_shape*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/sharding.py#L201-L208)[\#](#jax.sharding.Sharding.shard_shape "Link to this definition")  
Returns the shape of the data on each device.

The shard shape returned by this function is calculated from `global_shape` and the properties of the sharding.

Parameters:  
**global_shape** (*Shape*)

Return type:  
Shape

with_memory_kind(*kind*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/sharding.py#L137-L140)[\#](#jax.sharding.Sharding.with_memory_kind "Link to this definition")  
Returns a new Sharding instance with the specified memory kind.

Parameters:  
**kind** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)"))

Return type:  
[Sharding](#jax.sharding.Sharding "jax.sharding.Sharding")

&nbsp;

*class* jax.sharding.SingleDeviceSharding(*\*args*, *\*\*kwargs*)[\#](#jax.sharding.SingleDeviceSharding "Link to this definition")  
Bases: [`Sharding`](#jax.sharding.Sharding "jaxlib._jax.Sharding")

A [`Sharding`](#jax.sharding.Sharding "jax.sharding.Sharding") that places its data on a single device.

Parameters:  
**device** – A single `Device`.

Examples

    >>> single_device_sharding = jax.sharding.SingleDeviceSharding(
    ...     jax.devices()[0])

*property* device_set*: [set](https://docs.python.org/3/library/stdtypes.html#set "(in Python v3.14)")\[[Device](_autosummary/jax.Device.html#jax.Device "jax.Device")\]*[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/sharding_impls.py#L150-L153)[\#](#jax.sharding.SingleDeviceSharding.device_set "Link to this definition")  
The set of devices that this [`Sharding`](#jax.sharding.Sharding "jax.sharding.Sharding") spans.

In multi-controller JAX, the set of devices is global, i.e., includes non-addressable devices from other processes.

devices_indices_map(*global_shape*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/sharding_impls.py#L161-L163)[\#](#jax.sharding.SingleDeviceSharding.devices_indices_map "Link to this definition")  
Returns a mapping from devices to the array slices each contains.

The mapping includes all global devices, i.e., including non-addressable devices from other processes.

Parameters:  
**global_shape** (*Shape*)

Return type:  
Mapping\[[Device](_autosummary/jax.Device.html#jax.Device "jax.Device"), Index\]

*property* is_fully_addressable*: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")*[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/sharding_impls.py#L180-L183)[\#](#jax.sharding.SingleDeviceSharding.is_fully_addressable "Link to this definition")  
Is this sharding fully addressable?

A sharding is fully addressable if the current process can address all of the devices named in the [`Sharding`](#jax.sharding.Sharding "jax.sharding.Sharding"). `is_fully_addressable` is equivalent to “is_local” in multi-process JAX.

*property* is_fully_replicated*: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")*[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/sharding_impls.py#L176-L179)[\#](#jax.sharding.SingleDeviceSharding.is_fully_replicated "Link to this definition")  
Is this sharding fully replicated?

A sharding is fully replicated if each device has a complete copy of the entire data.

*property* memory_kind*: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)")*[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/sharding_impls.py#L154-L157)[\#](#jax.sharding.SingleDeviceSharding.memory_kind "Link to this definition")  
Returns the memory kind of the sharding.

*property* num_devices*: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/sharding_impls.py#L146-L149)[\#](#jax.sharding.SingleDeviceSharding.num_devices "Link to this definition")  
Number of devices that the sharding contains.

with_memory_kind(*kind*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/sharding_impls.py#L158-L160)[\#](#jax.sharding.SingleDeviceSharding.with_memory_kind "Link to this definition")  
Returns a new Sharding instance with the specified memory kind.

Parameters:  
**kind** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)"))

Return type:  
[SingleDeviceSharding](#jax.sharding.SingleDeviceSharding "jax.sharding.SingleDeviceSharding")

&nbsp;

*class* jax.sharding.NamedSharding(*\*args*, *\*\*kwargs*)[\#](#jax.sharding.NamedSharding "Link to this definition")  
Bases: [`Sharding`](#jax.sharding.Sharding "jaxlib._jax.Sharding")

A [`NamedSharding`](#jax.sharding.NamedSharding "jax.sharding.NamedSharding") expresses sharding using named axes.

A [`NamedSharding`](#jax.sharding.NamedSharding "jax.sharding.NamedSharding") is a pair of a [`Mesh`](#jax.sharding.Mesh "jax.sharding.Mesh") of devices and [`PartitionSpec`](#jax.sharding.PartitionSpec "jax.sharding.PartitionSpec") which describes how to shard an array across that mesh.

A [`Mesh`](#jax.sharding.Mesh "jax.sharding.Mesh") is a multidimensional NumPy array of JAX devices, where each axis of the mesh has a name, e.g. `'x'` or `'y'`.

A [`PartitionSpec`](#jax.sharding.PartitionSpec "jax.sharding.PartitionSpec") is a tuple, whose elements can be a `None`, a mesh axis, or a tuple of mesh axes. Each element describes how an input dimension is partitioned across zero or more mesh dimensions. For example, `PartitionSpec('x',`` ``'y')` says that the first dimension of data is sharded across `x` axis of the mesh, and the second dimension is sharded across `y` axis of the mesh.

The [Distributed arrays and automatic parallelization](https://docs.jax.dev/en/latest/parallel.html) and [Explicit Sharding](https://docs.jax.dev/en/latest/parallel.html) tutorials have more details and diagrams that explain how [`Mesh`](#jax.sharding.Mesh "jax.sharding.Mesh") and [`PartitionSpec`](#jax.sharding.PartitionSpec "jax.sharding.PartitionSpec") are used.

Parameters:  
- **mesh** – A [`jax.sharding.Mesh`](#jax.sharding.Mesh "jax.sharding.Mesh") object.

- **spec** – A [`jax.sharding.PartitionSpec`](#jax.sharding.PartitionSpec "jax.sharding.PartitionSpec") object.

- **memory_kind** – A string indicating the memory kind of the sharding.

Examples

    >>> from jax.sharding import Mesh
    >>> from jax.sharding import PartitionSpec as P
    >>> mesh = Mesh(np.array(jax.devices()).reshape(2, 4), ('x', 'y'))
    >>> spec = P('x', 'y')
    >>> named_sharding = jax.sharding.NamedSharding(mesh, spec)

*property* addressable_devices*: [set](https://docs.python.org/3/library/stdtypes.html#set "(in Python v3.14)")\[[Device](_autosummary/jax.Device.html#jax.Device "jax.Device")\]*[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/named_sharding.py#L199-L207)[\#](#jax.sharding.NamedSharding.addressable_devices "Link to this definition")  
The set of devices in the [`Sharding`](#jax.sharding.Sharding "jax.sharding.Sharding") that are addressable by the current process.

*property* device_set*: [set](https://docs.python.org/3/library/stdtypes.html#set "(in Python v3.14)")\[[Device](_autosummary/jax.Device.html#jax.Device "jax.Device")\]*[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/named_sharding.py#L171-L177)[\#](#jax.sharding.NamedSharding.device_set "Link to this definition")  
The set of devices that this [`Sharding`](#jax.sharding.Sharding "jax.sharding.Sharding") spans.

In multi-controller JAX, the set of devices is global, i.e., includes non-addressable devices from other processes.

is_equivalent_to(*other*, *ndim*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/named_sharding.py#L239-L245)[\#](#jax.sharding.NamedSharding.is_equivalent_to "Link to this definition")  
Returns `True` if two shardings are equivalent.

Two shardings are equivalent if they place the same logical array shards on the same devices.

Parameters:  
**ndim** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

Return type:  
[bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")

*property* is_fully_addressable*: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")*[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/named_sharding.py#L185-L192)[\#](#jax.sharding.NamedSharding.is_fully_addressable "Link to this definition")  
Is this sharding fully addressable?

A sharding is fully addressable if the current process can address all of the devices named in the [`Sharding`](#jax.sharding.Sharding "jax.sharding.Sharding"). `is_fully_addressable` is equivalent to “is_local” in multi-process JAX.

*property* is_fully_replicated*: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")*[\#](#jax.sharding.NamedSharding.is_fully_replicated "Link to this definition")  
Is this sharding fully replicated?

A sharding is fully replicated if each device has a complete copy of the entire data.

*property* memory_kind*: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)")*[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/named_sharding.py#L135-L138)[\#](#jax.sharding.NamedSharding.memory_kind "Link to this definition")  
Returns the memory kind of the sharding.

*property* mesh[\#](#jax.sharding.NamedSharding.mesh "Link to this definition")  
(self) -\> object

*property* num_devices*: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/named_sharding.py#L167-L170)[\#](#jax.sharding.NamedSharding.num_devices "Link to this definition")  
Number of devices that the sharding contains.

*property* spec[\#](#jax.sharding.NamedSharding.spec "Link to this definition")  
(self) -\> object

with_memory_kind(*kind*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/named_sharding.py#L225-L227)[\#](#jax.sharding.NamedSharding.with_memory_kind "Link to this definition")  
Returns a new Sharding instance with the specified memory kind.

Parameters:  
**kind** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)"))

Return type:  
[NamedSharding](#jax.sharding.NamedSharding "jax.sharding.NamedSharding")

&nbsp;

jax.sharding.PartitionSpec[\#](#jax.sharding.PartitionSpec "Link to this definition")  
alias of `P`

&nbsp;

*class* jax.sharding.Mesh(*devices*, *axis_names*, *axis_types=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/mesh.py#L216-L425)[\#](#jax.sharding.Mesh "Link to this definition")  
Declare the hardware resources available in the scope of this manager.

See [Distributed arrays and automatic parallelization](https://docs.jax.dev/en/latest/parallel.html) and [Explicit Sharding](https://docs.jax.dev/en/latest/parallel.html) tutorials.

Parameters:  
- **devices** (*np.ndarray*) – A NumPy ndarray object containing JAX device objects (as obtained e.g. from [`jax.devices()`](_autosummary/jax.devices.html#jax.devices "jax.devices")).

- **axis_names** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[MeshAxisName,* *...\]*) – A sequence of resource axis names to be assigned to the dimensions of the `devices` argument. Its length should match the rank of `devices`.

- **axis_types** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[AxisType,* *...\]*) – and optional tuple of `jax.sharding.AxisType` entries corresponding to the `axis_names`. See [Explicit Sharding](https://docs.jax.dev/en/latest/parallel.html) for more information.

Examples

    >>> from jax.sharding import Mesh
    >>> from jax.sharding import PartitionSpec as P, NamedSharding
    >>> import numpy as np
    ...
    >>> # Declare a 2D mesh with axes `x` and `y`.
    >>> devices = np.array(jax.devices()).reshape(4, 2)
    >>> mesh = Mesh(devices, ('x', 'y'))
    >>> inp = np.arange(16).reshape(8, 2)
    >>> arr = jax.device_put(inp, NamedSharding(mesh, P('x', 'y')))
    >>> out = jax.jit(lambda x: x * 2)(arr)
    >>> assert out.sharding == NamedSharding(mesh, P('x', 'y'))

[](_autosummary/jax.random.weibull_min.html "previous page")

previous

jax.random.weibull_min

[](jax.ad_checkpoint.html "next page")

next

`jax.ad_checkpoint` module

Contents

- [Classes](#classes)
  - [`Sharding`](#jax.sharding.Sharding)
    - [`Sharding.addressable_devices`](#jax.sharding.Sharding.addressable_devices)
    - [`Sharding.addressable_devices_indices_map()`](#jax.sharding.Sharding.addressable_devices_indices_map)
    - [`Sharding.device_set`](#jax.sharding.Sharding.device_set)
    - [`Sharding.devices_indices_map()`](#jax.sharding.Sharding.devices_indices_map)
    - [`Sharding.is_equivalent_to()`](#jax.sharding.Sharding.is_equivalent_to)
    - [`Sharding.is_fully_addressable`](#jax.sharding.Sharding.is_fully_addressable)
    - [`Sharding.is_fully_replicated`](#jax.sharding.Sharding.is_fully_replicated)
    - [`Sharding.memory_kind`](#jax.sharding.Sharding.memory_kind)
    - [`Sharding.num_devices`](#jax.sharding.Sharding.num_devices)
    - [`Sharding.shard_shape()`](#jax.sharding.Sharding.shard_shape)
    - [`Sharding.with_memory_kind()`](#jax.sharding.Sharding.with_memory_kind)
  - [`SingleDeviceSharding`](#jax.sharding.SingleDeviceSharding)
    - [`SingleDeviceSharding.device_set`](#jax.sharding.SingleDeviceSharding.device_set)
    - [`SingleDeviceSharding.devices_indices_map()`](#jax.sharding.SingleDeviceSharding.devices_indices_map)
    - [`SingleDeviceSharding.is_fully_addressable`](#jax.sharding.SingleDeviceSharding.is_fully_addressable)
    - [`SingleDeviceSharding.is_fully_replicated`](#jax.sharding.SingleDeviceSharding.is_fully_replicated)
    - [`SingleDeviceSharding.memory_kind`](#jax.sharding.SingleDeviceSharding.memory_kind)
    - [`SingleDeviceSharding.num_devices`](#jax.sharding.SingleDeviceSharding.num_devices)
    - [`SingleDeviceSharding.with_memory_kind()`](#jax.sharding.SingleDeviceSharding.with_memory_kind)
  - [`NamedSharding`](#jax.sharding.NamedSharding)
    - [`NamedSharding.addressable_devices`](#jax.sharding.NamedSharding.addressable_devices)
    - [`NamedSharding.device_set`](#jax.sharding.NamedSharding.device_set)
    - [`NamedSharding.is_equivalent_to()`](#jax.sharding.NamedSharding.is_equivalent_to)
    - [`NamedSharding.is_fully_addressable`](#jax.sharding.NamedSharding.is_fully_addressable)
    - [`NamedSharding.is_fully_replicated`](#jax.sharding.NamedSharding.is_fully_replicated)
    - [`NamedSharding.memory_kind`](#jax.sharding.NamedSharding.memory_kind)
    - [`NamedSharding.mesh`](#jax.sharding.NamedSharding.mesh)
    - [`NamedSharding.num_devices`](#jax.sharding.NamedSharding.num_devices)
    - [`NamedSharding.spec`](#jax.sharding.NamedSharding.spec)
    - [`NamedSharding.with_memory_kind()`](#jax.sharding.NamedSharding.with_memory_kind)
  - [`PartitionSpec`](#jax.sharding.PartitionSpec)
  - [`Mesh`](#jax.sharding.Mesh)

By The JAX authors

© Copyright 2024, The JAX Authors.\
