- [](../index.html)
- [API Reference](../jax.html)
- jax.shard_map

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.shard_map.rst "Download source file")
-  .pdf

# jax.shard_map

## Contents

- [`shard_map()`](#jax.shard_map)

# jax.shard_map[\#](#jax-shard-map "Link to this heading")

jax.shard_map(*f: F*, */*, *\**, *out_specs: Specs*, *in_specs: Specs \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") \| InferFromArgs = jax.sharding.Infer*, *mesh: [Mesh](../jax.sharding.html#jax.sharding.Mesh "jax.sharding.Mesh") \| AbstractMesh \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *axis_names: Set\[AxisName\] = frozenset({})*, *check_vma: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = True*) → F[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/shard_map.py#L107-L164)[\#](#jax.shard_map "Link to this definition")\
jax.shard_map(*f: [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, */*, *\**, *out_specs: Specs*, *in_specs: Specs \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") \| InferFromArgs = jax.sharding.Infer*, *mesh: [Mesh](../jax.sharding.html#jax.sharding.Mesh "jax.sharding.Mesh") \| AbstractMesh \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *axis_names: Set\[AxisName\] = frozenset({})*, *check_vma: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = True*) → [Callable](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")\[\[G\], G\]  
Map a function over shards of data using a mesh of devices.

See the docs at [https://docs.jax.dev/en/latest/notebooks/shard_map.html](https://docs.jax.dev/en/latest/notebooks/shard_map.html).

Parameters:  
- **f** – callable to be mapped. Each application of `f`, or “instance” of `f`, takes as input a shard of the mapped-over arguments and produces a shard of the output.

- **mesh** – (optional, default None) a `jax.sharding.Mesh` representing the array of devices over which to shard the data and on which to execute instances of `f`. The names of the `Mesh` can be used in collective communication operations in `f`. If mesh is None, it will be inferred from the context which can be set via jax.set_mesh context manager.

- **in_specs** – (optional, default Infer) a pytree with `jax.sharding.PartitionSpec` instances as leaves, with a tree structure that is a tree prefix of the args tuple to be mapped over. Similar to `jax.sharding.NamedSharding`, each `PartitionSpec` represents how the corresponding argument (or subtree of arguments) should be sharded along the named axes of `mesh`. In each `PartitionSpec`, mentioning a `mesh` axis name at a position expresses sharding the corresponding argument array axis along that positional axis; not mentioning an axis name expresses replication. If `Infer`, all mesh axes must be of type Explicit, in which case the in_specs are inferred from the argument types. If `None`, inputs will be treated as static.

- **out_specs** – a pytree with `PartitionSpec` instances as leaves, with a tree structure that is a tree prefix of the output of `f`. Each `PartitionSpec` represents how the corresponding output shards should be concatenated. In each `PartitionSpec`, mentioning a `mesh` axis name at a position expresses concatenation of that mesh axis’s shards along the corresponding positional axis; not mentioning a `mesh` axis name expresses a promise that the output values are equal along that mesh axis, and that rather than concatenating only a single value should be produced.

- **axis_names** – (optional, default set()) set of axis names from `mesh` over which the function `f` is manual. If empty, `f`, is manual over all mesh axes.

- **check_vma** – (optional) boolean (default True) representing whether to enable additional validity checks and automatic differentiation optimizations. The validity checks concern whether any mesh axis names not mentioned in `out_specs` are consistent with how the outputs of `f` are replicated.

Returns:  
A callable representing a mapped version of `f`, which accepts positional arguments corresponding to those of `f` and produces output corresponding to that of `f`.

[](jax.vmap.html "previous page")

previous

jax.vmap

[](jax.smap.html "next page")

next

jax.smap

Contents

- [`shard_map()`](#jax.shard_map)

By The JAX authors

© Copyright 2024, The JAX Authors.\
