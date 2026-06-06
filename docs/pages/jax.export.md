- [](index.html)
- [API Reference](jax.html)
- `jax.export` module

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](_sources/jax.export.rst "Download source file")
-  .pdf

# jax.export module

## Contents

- [Classes](#classes)
  - [`Exported`](#jax.export.Exported)
    - [`Exported.fun_name`](#jax.export.Exported.fun_name)
    - [`Exported.in_tree`](#jax.export.Exported.in_tree)
    - [`Exported.in_avals`](#jax.export.Exported.in_avals)
    - [`Exported.out_tree`](#jax.export.Exported.out_tree)
    - [`Exported.out_avals`](#jax.export.Exported.out_avals)
    - [`Exported.in_shardings_hlo`](#jax.export.Exported.in_shardings_hlo)
    - [`Exported.out_shardings_hlo`](#jax.export.Exported.out_shardings_hlo)
    - [`Exported.nr_devices`](#jax.export.Exported.nr_devices)
    - [`Exported.platforms`](#jax.export.Exported.platforms)
    - [`Exported.ordered_effects`](#jax.export.Exported.ordered_effects)
    - [`Exported.unordered_effects`](#jax.export.Exported.unordered_effects)
    - [`Exported.mlir_module_serialized`](#jax.export.Exported.mlir_module_serialized)
    - [`Exported.calling_convention_version`](#jax.export.Exported.calling_convention_version)
    - [`Exported.module_kept_var_idx`](#jax.export.Exported.module_kept_var_idx)
    - [`Exported.uses_global_constants`](#jax.export.Exported.uses_global_constants)
    - [`Exported.disabled_safety_checks`](#jax.export.Exported.disabled_safety_checks)
    - [`Exported._get_vjp`](#jax.export.Exported._get_vjp)
    - [`Exported.call()`](#jax.export.Exported.call)
    - [`Exported.has_vjp()`](#jax.export.Exported.has_vjp)
    - [`Exported.in_shardings_jax()`](#jax.export.Exported.in_shardings_jax)
    - [`Exported.mlir_module()`](#jax.export.Exported.mlir_module)
    - [`Exported.out_shardings_jax()`](#jax.export.Exported.out_shardings_jax)
    - [`Exported.serialize()`](#jax.export.Exported.serialize)
    - [`Exported.vjp()`](#jax.export.Exported.vjp)
  - [`DisabledSafetyCheck`](#jax.export.DisabledSafetyCheck)
    - [`DisabledSafetyCheck.custom_call()`](#jax.export.DisabledSafetyCheck.custom_call)
    - [`DisabledSafetyCheck.is_custom_call()`](#jax.export.DisabledSafetyCheck.is_custom_call)
    - [`DisabledSafetyCheck.platform()`](#jax.export.DisabledSafetyCheck.platform)
- [Functions](#functions)
- [Functions related to shape polymorphism](#functions-related-to-shape-polymorphism)
- [Constants](#constants)
  - [`jax.export.minimum_supported_serialization_version`](#jax.export.jax.export.minimum_supported_serialization_version)
  - [`jax.export.maximum_supported_serialization_version`](#jax.export.jax.export.maximum_supported_serialization_version)

# `jax.export` module[\#](#module-jax.export "Link to this heading")

[`jax.export`](#module-jax.export "jax.export") is a library for exporting and serializing JAX functions for persistent archival.

See the [Exporting and serialization](export/index.html#export) documentation.

## Classes[\#](#classes "Link to this heading")

*class* jax.export.Exported(*fun_name*, *in_tree*, *in_avals*, *out_tree*, *out_avals*, *\_has_named_shardings*, *\_in_named_shardings*, *\_out_named_shardings*, *in_shardings_hlo*, *out_shardings_hlo*, *nr_devices*, *platforms*, *ordered_effects*, *unordered_effects*, *disabled_safety_checks*, *mlir_module_serialized*, *calling_convention_version*, *module_kept_var_idx*, *uses_global_constants*, *\_get_vjp*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/export/_export.py#L130-L347)[\#](#jax.export.Exported "Link to this definition")  
A JAX function lowered to StableHLO.

Parameters:  
- **fun_name** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)"))

- **in_tree** (*tree_util.PyTreeDef*)

- **in_avals** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[core.ShapedArray,* *...\]*)

- **out_tree** (*tree_util.PyTreeDef*)

- **out_avals** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[core.ShapedArray,* *...\]*)

- **\_has_named_shardings** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

- **\_in_named_shardings** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*NamedSharding*](jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding") *\|* *None,* *...\]*)

- **\_out_named_shardings** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*NamedSharding*](jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding") *\|* *None,* *...\]*)

- **in_shardings_hlo** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[HloSharding* *\|* *None,* *...\]*)

- **out_shardings_hlo** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[HloSharding* *\|* *None,* *...\]*)

- **nr_devices** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **platforms** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*,* *...\]*)

- **ordered_effects** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*effects.Effect*](_autosummary/jax.extend.core.Effect.html#jax.extend.core.Effect "jax._src.effects.Effect")*,* *...\]*)

- **unordered_effects** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*effects.Effect*](_autosummary/jax.extend.core.Effect.html#jax.extend.core.Effect "jax._src.effects.Effect")*,* *...\]*)

- **disabled_safety_checks** (*Sequence\[*[*DisabledSafetyCheck*](#jax.export.DisabledSafetyCheck "jax.export.DisabledSafetyCheck")*\]*)

- **mlir_module_serialized** ([*bytes*](https://docs.python.org/3/library/stdtypes.html#bytes "(in Python v3.14)"))

- **calling_convention_version** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **module_kept_var_idx** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *...\]*)

- **uses_global_constants** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

- **\_get_vjp** ([*Callable*](_autosummary/jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[\[*[*Exported*](#jax.export.Exported "jax.export.Exported")*\],* [*Exported*](#jax.export.Exported "jax.export.Exported")*\]* *\|* *None*)

fun_name[\#](#jax.export.Exported.fun_name "Link to this definition")  
the name of the exported function, for error messages.

Type:  
[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")

in_tree[\#](#jax.export.Exported.in_tree "Link to this definition")  
a PyTreeDef describing the tuple (args, kwargs) of the lowered JAX function. The actual lowering does not depend on the `in_tree`, but this can be used to invoke the exported function using the same argument structure.

Type:  
tree_util.PyTreeDef

in_avals[\#](#jax.export.Exported.in_avals "Link to this definition")  
the flat tuple of input abstract values. May contain dimension expressions in the shapes.

Type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[core.ShapedArray, …\]

out_tree[\#](#jax.export.Exported.out_tree "Link to this definition")  
a PyTreeDef describing the result of the lowered JAX function.

Type:  
tree_util.PyTreeDef

out_avals[\#](#jax.export.Exported.out_avals "Link to this definition")  
the flat tuple of output abstract values. May contain dimension expressions in the shapes, with dimension variables among those in `in_avals`. Note that when the out_shardings are not specified for an output, the out_avals.sharding.spec for Auto axes may be None even if after compilation the compiler may pick a non-replicated sharding. See [https://docs.jax.dev/en/latest/parallel.html](https://docs.jax.dev/en/latest/parallel.html) for more details.

Type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[core.ShapedArray, …\]

in_shardings_hlo[\#](#jax.export.Exported.in_shardings_hlo "Link to this definition")  
(Not used for exports created after 3/17/2026.) the flattened input shardings, a sequence as long as `in_avals`. `None` means unspecified sharding. Note that these do not include the mesh or the actual devices used in the mesh, and in general you should avoid using this field directly. See `in_shardings_jax` for a way to turn these into sharding specification that can be used with JAX APIs.

Type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[HloSharding \| None, …\]

out_shardings_hlo[\#](#jax.export.Exported.out_shardings_hlo "Link to this definition")  
(Not used for exports created after 3/17/2026.) the flattened output shardings, a sequence as long as `out_avals`. `None` means unspecified sharding. Note that these do not include the mesh or the actual devices used in the mesh, and in general you should avoid using this field directly. See `out_shardings_jax` for a way to turn these into sharding specification that can be used with JAX APIs.

Type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[HloSharding \| None, …\]

nr_devices[\#](#jax.export.Exported.nr_devices "Link to this definition")  
the number of devices that the module has been lowered for.

Type:  
[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

platforms[\#](#jax.export.Exported.platforms "Link to this definition")  
a tuple containing the platforms for which the function should be exported. The set of platforms in JAX is open-ended; users can add platforms. JAX built-in platforms are: ‘tpu’, ‘cpu’, ‘cuda’, ‘rocm’. See [https://docs.jax.dev/en/latest/export/export.html#cross-platform-and-multi-platform-export](https://docs.jax.dev/en/latest/export/export.html#cross-platform-and-multi-platform-export).

Type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)"), …\]

ordered_effects[\#](#jax.export.Exported.ordered_effects "Link to this definition")  
the ordered effects present in the serialized module. This is present from serialization version 9. See [https://docs.jax.dev/en/latest/export/export.html#module-calling-convention](https://docs.jax.dev/en/latest/export/export.html#module-calling-convention) for the calling convention in presence of ordered effects.

Type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[effects.Effect](_autosummary/jax.extend.core.Effect.html#jax.extend.core.Effect "jax._src.effects.Effect"), …\]

unordered_effects[\#](#jax.export.Exported.unordered_effects "Link to this definition")  
the unordered effects present in the serialized module. This is present from serialization version 9.

Type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[effects.Effect](_autosummary/jax.extend.core.Effect.html#jax.extend.core.Effect "jax._src.effects.Effect"), …\]

mlir_module_serialized[\#](#jax.export.Exported.mlir_module_serialized "Link to this definition")  
the serialized lowered VHLO module.

Type:  
[bytes](https://docs.python.org/3/library/stdtypes.html#bytes "(in Python v3.14)")

calling_convention_version[\#](#jax.export.Exported.calling_convention_version "Link to this definition")  
a version number for the calling convention of the exported module. See more versioning details at [https://docs.jax.dev/en/latest/export/export.html#calling-convention-versions](https://docs.jax.dev/en/latest/export/export.html#calling-convention-versions).

Type:  
[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

module_kept_var_idx[\#](#jax.export.Exported.module_kept_var_idx "Link to this definition")  
the sorted indices of the arguments among in_avals that must be passed to the module. The other arguments have been dropped because they are not used.

Type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"), …\]

uses_global_constants[\#](#jax.export.Exported.uses_global_constants "Link to this definition")  
whether the `mlir_module_serialized` uses shape polymorphism or multi-platform export. This may be because `in_avals` contains dimension variables, or due to inner calls of Exported modules that have dimension variables or platform index arguments. Such modules need shape refinement before XLA compilation.

Type:  
[bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")

disabled_safety_checks[\#](#jax.export.Exported.disabled_safety_checks "Link to this definition")  
a list of descriptors of safety checks that have been disabled at export time. See docstring for `DisabledSafetyCheck`.

Type:  
Sequence\[[DisabledSafetyCheck](#jax.export.DisabledSafetyCheck "jax.export.DisabledSafetyCheck")\]

\_get_vjp[\#](#jax.export.Exported._get_vjp "Link to this definition")  
an optional function that takes the current exported function and returns the exported VJP function. The VJP function takes a flat list of arguments, starting with the primal arguments and followed by a cotangent argument for each primal output. It returns a tuple with the cotangents corresponding to the flattened primal inputs.

Type:  
[Callable](_autosummary/jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")\[\[[Exported](#jax.export.Exported "jax.export.Exported")\], [Exported](#jax.export.Exported "jax.export.Exported")\] \| None

DO NOT RELY directly on fields whose name starts with ‘\_’. They will change.

See a description of the calling convention for the [`mlir_module()`](#jax.export.Exported.mlir_module "jax.export.Exported.mlir_module") method at [https://docs.jax.dev/en/latest/export/export.html#module-calling-convention](https://docs.jax.dev/en/latest/export/export.html#module-calling-convention).

call(*\*args*, *\*\*kwargs*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/export/_export.py#L330-L347)[\#](#jax.export.Exported.call "Link to this definition")  
Call an exported function from a JAX program.

Parameters:  
- **args** – the positional arguments to pass to the exported function. This should be a pytree of arrays with the same pytree structure as the arguments for which the function was exported.

- **kwargs** – the keyword arguments to pass to the exported function.

Returns: a pytree of result array, with the same structure as the  
results of the exported function.

The invocation supports reverse-mode AD, and all the features supported by exporting: shape polymorphism, multi-platform, device polymorphism. See the examples in the \[JAX export documentation\]([https://docs.jax.dev/en/latest/export/export.html](https://docs.jax.dev/en/latest/export/export.html)).

has_vjp()[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/export/_export.py#L301-L304)[\#](#jax.export.Exported.has_vjp "Link to this definition")  
Returns if this Exported supports VJP.

Return type:  
[bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")

in_shardings_jax(*mesh*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/export/_export.py#L244-L287)[\#](#jax.export.Exported.in_shardings_jax "Link to this definition")  
Creates Shardings corresponding to `self.in_shardings_hlo` and `self._in_named_shardings`.

The Exported object stores `in_shardings_hlo` as HloShardings, and after 12/5/2025 also `_in_named_shardings` as NamedShardings with abstract meshes. This method constructs Sharding that can be used in JAX APIs such as [`jax.jit()`](_autosummary/jax.jit.html#jax.jit "jax.jit") or [`jax.device_put()`](_autosummary/jax.device_put.html#jax.device_put "jax.device_put"). The mesh argument may be a concrete mesh.

Example usage:

    >>> from jax import export, sharding
    >>> # Prepare the exported object:
    >>> exp_mesh = sharding.Mesh(jax.devices(), ("a",))
    >>> exp = export.export(jax.jit(lambda x: jax.numpy.add(x, x),
    ...                             in_shardings=sharding.NamedSharding(exp_mesh, sharding.PartitionSpec("a")))
    ...     )(np.arange(jax.device_count()))
    >>> exp.in_shardings_jax(exp_mesh)
    (NamedSharding(mesh=Mesh('a': 8, axis_types=(Auto,)), spec=P('a',), memory_kind=device),)
    >>> # Create a mesh for running the exported object
    >>> run_mesh = sharding.Mesh(jax.devices()[::-1], ("a",))
    >>> # Put the args and kwargs on the appropriate devices
    >>> run_arg = jax.device_put(np.arange(jax.device_count()),
    ...     exp.in_shardings_jax(run_mesh)[0])
    >>> res = exp.call(run_arg)
    >>> res.addressable_shards
    [Shard(device=CpuDevice(id=7), index=(slice(0, 1, None),), replica_id=0, data=[0]),
     Shard(device=CpuDevice(id=6), index=(slice(1, 2, None),), replica_id=0, data=[2]),
     Shard(device=CpuDevice(id=5), index=(slice(2, 3, None),), replica_id=0, data=[4]),
     Shard(device=CpuDevice(id=4), index=(slice(3, 4, None),), replica_id=0, data=[6]),
     Shard(device=CpuDevice(id=3), index=(slice(4, 5, None),), replica_id=0, data=[8]),
     Shard(device=CpuDevice(id=2), index=(slice(5, 6, None),), replica_id=0, data=[10]),
     Shard(device=CpuDevice(id=1), index=(slice(6, 7, None),), replica_id=0, data=[12]),
     Shard(device=CpuDevice(id=0), index=(slice(7, 8, None),), replica_id=0, data=[14])]

Parameters:  
**mesh** (*mesh_lib.Mesh* *\|* *mesh_lib.AbstractMesh*)

Return type:  
Sequence\[[sharding.Sharding](jax.sharding.html#jax.sharding.Sharding "jax.sharding.Sharding") \| None\]

mlir_module(*serialized=True*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/export/_export.py#L230-L238)[\#](#jax.export.Exported.mlir_module "Link to this definition")  
A string or Module representation of the `mlir_module_serialized`.

Parameters:  
**serialized** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

Return type:  
Any

out_shardings_jax(*mesh*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/export/_export.py#L288-L300)[\#](#jax.export.Exported.out_shardings_jax "Link to this definition")  
Creates Shardings for `out_shardings_hlo` and `_out_named_shardings`.

See documentation for in_shardings_jax.

Parameters:  
**mesh** (*mesh_lib.Mesh* *\|* *mesh_lib.AbstractMesh*)

Return type:  
Sequence\[[sharding.Sharding](jax.sharding.html#jax.sharding.Sharding "jax.sharding.Sharding") \| None\]

serialize(*vjp_order=0*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/export/_export.py#L315-L329)[\#](#jax.export.Exported.serialize "Link to this definition")  
Serializes an Exported.

Parameters:  
**vjp_order** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – The maximum vjp order to include. E.g., the value 2 means that we serialize the primal functions and two orders of the `vjp` function. This should allow 2nd order reverse mode differentiation of the deserialized function. i.e., `jax.grad(jax.grad(f))`.

Return type:  
[bytearray](https://docs.python.org/3/library/stdtypes.html#bytearray "(in Python v3.14)")

vjp()[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/export/_export.py#L305-L314)[\#](#jax.export.Exported.vjp "Link to this definition")  
Gets the exported VJP.

Returns None if not available, which can happen if the Exported has been loaded from an external format without a VJP.

Return type:  
[Exported](#jax.export.Exported "jax.export.Exported")

&nbsp;

*class* jax.export.DisabledSafetyCheck(*\_impl*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/export/_export.py#L79-L128)[\#](#jax.export.DisabledSafetyCheck "Link to this definition")  
A safety check that should be skipped on (de)serialization.

Most of these checks are performed on serialization, but some are deferred to deserialization. The list of disabled checks is attached to the serialization, e.g., as a sequence of string attributes to [`jax.export.Exported`](#jax.export.Exported "jax.export.Exported") or of `tf.XlaCallModuleOp`.

When using jax2tf, you can disable more deserialization safety checks by passing `TF_XLA_FLAGS=--tf_xla_call_module_disabled_checks=platform`.

Parameters:  
**\_impl** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)"))

*classmethod* custom_call(*target_name*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/export/_export.py#L100-L109)[\#](#jax.export.DisabledSafetyCheck.custom_call "Link to this definition")  
Allows the serialization of a call target not known to be stable.

Has effect only on serialization. :param target_name: the name of the custom call target to allow.

Parameters:  
**target_name** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)"))

Return type:  
[DisabledSafetyCheck](#jax.export.DisabledSafetyCheck "jax.export.DisabledSafetyCheck")

is_custom_call()[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/export/_export.py#L110-L114)[\#](#jax.export.DisabledSafetyCheck.is_custom_call "Link to this definition")  
Returns the custom call target allowed by this directive.

Return type:  
[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") \| None

*classmethod* platform()[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/export/_export.py#L92-L99)[\#](#jax.export.DisabledSafetyCheck.platform "Link to this definition")  
Allows the compilation platform to differ from the export platform.

Has effect only on deserialization.

Return type:  
[DisabledSafetyCheck](#jax.export.DisabledSafetyCheck "jax.export.DisabledSafetyCheck")

## Functions[\#](#functions "Link to this heading")

|  |  |
|----|----|
| [`export`](_autosummary/jax.export.export.html#jax.export.export "jax.export.export")(fun_jit, \*\[, platforms, ...\]) | Exports a JAX function for persistent serialization. |
| [`deserialize`](_autosummary/jax.export.deserialize.html#jax.export.deserialize "jax.export.deserialize")(blob) | Deserializes an Exported. |
| [`minimum_supported_calling_convention_version`](_autosummary/jax.export.minimum_supported_calling_convention_version.html#jax.export.minimum_supported_calling_convention_version "jax.export.minimum_supported_calling_convention_version") | int(\[x\]) -\> integer int(x, base=10) -\> integer |
| [`maximum_supported_calling_convention_version`](_autosummary/jax.export.maximum_supported_calling_convention_version.html#jax.export.maximum_supported_calling_convention_version "jax.export.maximum_supported_calling_convention_version") | int(\[x\]) -\> integer int(x, base=10) -\> integer |
| [`default_export_platform`](_autosummary/jax.export.default_export_platform.html#jax.export.default_export_platform "jax.export.default_export_platform")() | Retrieves the default export platform. |
| [`register_pytree_node_serialization`](_autosummary/jax.export.register_pytree_node_serialization.html#jax.export.register_pytree_node_serialization "jax.export.register_pytree_node_serialization")(nodetype, ...) | Registers a custom PyTree node for serialization and deserialization. |
| [`register_namedtuple_serialization`](_autosummary/jax.export.register_namedtuple_serialization.html#jax.export.register_namedtuple_serialization "jax.export.register_namedtuple_serialization")(nodetype, ...) | Registers a namedtuple for serialization and deserialization. |

## Functions related to shape polymorphism[\#](#functions-related-to-shape-polymorphism "Link to this heading")

|  |  |
|----|----|
| [`symbolic_shape`](_autosummary/jax.export.symbolic_shape.html#jax.export.symbolic_shape "jax.export.symbolic_shape")(shape_spec, \*\[, constraints, ...\]) | Constructs a symbolic shape from a string representation. |
| [`symbolic_args_specs`](_autosummary/jax.export.symbolic_args_specs.html#jax.export.symbolic_args_specs "jax.export.symbolic_args_specs")(args, shapes_specs\[, ...\]) | Constructs a pytree of jax.ShapeDtypeStruct arguments specs for export. |
| [`is_symbolic_dim`](_autosummary/jax.export.is_symbolic_dim.html#jax.export.is_symbolic_dim "jax.export.is_symbolic_dim")(p) | Checks if a dimension is symbolic. |
| [`SymbolicScope`](_autosummary/jax.export.SymbolicScope.html#jax.export.SymbolicScope "jax.export.SymbolicScope")(\[constraints_str\]) | Identifies a scope for symbolic expressions. |

## Constants[\#](#constants "Link to this heading")

jax.export.minimum_supported_serialization_version[\#](#jax.export.jax.export.minimum_supported_serialization_version "Link to this definition")  
The minimum supported serialization version; see [Calling convention versions](export/export.html#export-calling-convention-version).

&nbsp;

jax.export.maximum_supported_serialization_version[\#](#jax.export.jax.export.maximum_supported_serialization_version "Link to this definition")  
The maximum supported serialization version; see [Calling convention versions](export/export.html#export-calling-convention-version).

[](_autosummary/jax.typing.DTypeLike.html "previous page")

previous

jax.typing.DTypeLike

[](_autosummary/jax.export.export.html "next page")

next

jax.export.export

Contents

- [Classes](#classes)
  - [`Exported`](#jax.export.Exported)
    - [`Exported.fun_name`](#jax.export.Exported.fun_name)
    - [`Exported.in_tree`](#jax.export.Exported.in_tree)
    - [`Exported.in_avals`](#jax.export.Exported.in_avals)
    - [`Exported.out_tree`](#jax.export.Exported.out_tree)
    - [`Exported.out_avals`](#jax.export.Exported.out_avals)
    - [`Exported.in_shardings_hlo`](#jax.export.Exported.in_shardings_hlo)
    - [`Exported.out_shardings_hlo`](#jax.export.Exported.out_shardings_hlo)
    - [`Exported.nr_devices`](#jax.export.Exported.nr_devices)
    - [`Exported.platforms`](#jax.export.Exported.platforms)
    - [`Exported.ordered_effects`](#jax.export.Exported.ordered_effects)
    - [`Exported.unordered_effects`](#jax.export.Exported.unordered_effects)
    - [`Exported.mlir_module_serialized`](#jax.export.Exported.mlir_module_serialized)
    - [`Exported.calling_convention_version`](#jax.export.Exported.calling_convention_version)
    - [`Exported.module_kept_var_idx`](#jax.export.Exported.module_kept_var_idx)
    - [`Exported.uses_global_constants`](#jax.export.Exported.uses_global_constants)
    - [`Exported.disabled_safety_checks`](#jax.export.Exported.disabled_safety_checks)
    - [`Exported._get_vjp`](#jax.export.Exported._get_vjp)
    - [`Exported.call()`](#jax.export.Exported.call)
    - [`Exported.has_vjp()`](#jax.export.Exported.has_vjp)
    - [`Exported.in_shardings_jax()`](#jax.export.Exported.in_shardings_jax)
    - [`Exported.mlir_module()`](#jax.export.Exported.mlir_module)
    - [`Exported.out_shardings_jax()`](#jax.export.Exported.out_shardings_jax)
    - [`Exported.serialize()`](#jax.export.Exported.serialize)
    - [`Exported.vjp()`](#jax.export.Exported.vjp)
  - [`DisabledSafetyCheck`](#jax.export.DisabledSafetyCheck)
    - [`DisabledSafetyCheck.custom_call()`](#jax.export.DisabledSafetyCheck.custom_call)
    - [`DisabledSafetyCheck.is_custom_call()`](#jax.export.DisabledSafetyCheck.is_custom_call)
    - [`DisabledSafetyCheck.platform()`](#jax.export.DisabledSafetyCheck.platform)
- [Functions](#functions)
- [Functions related to shape polymorphism](#functions-related-to-shape-polymorphism)
- [Constants](#constants)
  - [`jax.export.minimum_supported_serialization_version`](#jax.export.jax.export.minimum_supported_serialization_version)
  - [`jax.export.maximum_supported_serialization_version`](#jax.export.jax.export.maximum_supported_serialization_version)

By The JAX authors

© Copyright 2024, The JAX Authors.\
