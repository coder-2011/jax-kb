- [](../index.html)
- [API Reference](../jax.html)
- [`jax.extend` module](../jax.extend.html)
- [`jax.extend.xla` module](../jax.extend.xla.html)
- jax.extend.xla.register_hlo_module_transformation

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.extend.xla.register_hlo_module_transformation.rst "Download source file")
-  .pdf

# jax.extend.xla.register_hlo_module_transformation

## Contents

- [`register_hlo_module_transformation()`](#jax.extend.xla.register_hlo_module_transformation)

# jax.extend.xla.register_hlo_module_transformation[\#](#jax-extend-xla-register-hlo-module-transformation "Link to this heading")

jax.extend.xla.register_hlo_module_transformation(*callback*, *\**, *name*, *stage=PipelineStage.PRE_SCHEDULER*, *platforms=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/xla_transform.py#L34-L99)[\#](#jax.extend.xla.register_hlo_module_transformation "Link to this definition")  
Register a custom compiler pass that transforms HLO modules.

The registered pass will be called during XLA compilation at the specified pipeline stage. The callback receives a serialized `HloModuleProto` as bytes and should return either:

- Modified serialized `HloModuleProto` bytes if the module was changed.

- `None` if no changes were made.

Multiple registration calls at the same stage (with different callbacks) will be added to a queue, and be invoked in the order they were registered.

Parameters:  
- **callback** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[\[*[*bytes*](https://docs.python.org/3/library/stdtypes.html#bytes "(in Python v3.14)")*\],* [*bytes*](https://docs.python.org/3/library/stdtypes.html#bytes "(in Python v3.14)") *\|* *None\]*) – A function `(bytes)`` ``->`` ``bytes`` ``|`` ``None` that receives a serialized HloModuleProto and optionally returns a modified one.

- **name** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – A name for the compiler pass.

- **stage** ([*PipelineStage*](jax.extend.xla.PipelineStage.html#jax.extend.xla.PipelineStage "jax.extend.xla.PipelineStage")) – The pipeline stage at which the pass runs. Must be a `PipelineStage` enum.

- **platforms** (*Sequence\[*[*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*\]* *\|* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – The list of platforms to register the pass for (e.g. `"cpu"`, `"tpu"`). If `None`, the pass is registered for all known backends by default. Can be a single platform string or a sequence of strings.

Return type:  
None

[](jax.extend.xla.PipelineStage.html "previous page")

previous

jax.extend.xla.PipelineStage

[](../jax.example_libraries.html "next page")

next

`jax.example_libraries` module

Contents

- [`register_hlo_module_transformation()`](#jax.extend.xla.register_hlo_module_transformation)

By The JAX authors

© Copyright 2024, The JAX Authors.\
