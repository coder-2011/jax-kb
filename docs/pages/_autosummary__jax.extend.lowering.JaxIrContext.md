- [](../index.html)
- [API Reference](../jax.html)
- [`jax.extend` module](../jax.extend.html)
- [`jax.extend.lowering` module](../jax.extend.lowering.html)
- jax.extend.lowering.JaxIrContext

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.extend.lowering.JaxIrContext.rst "Download source file")
-  .pdf

# jax.extend.lowering.JaxIrContext

## Contents

- [`JaxIrContext`](#jax.extend.lowering.JaxIrContext)
  - [`JaxIrContext.__init__()`](#jax.extend.lowering.JaxIrContext.__init__)

# jax.extend.lowering.JaxIrContext[\#](#jax-extend-lowering-jaxircontext "Link to this heading")

*class* jax.extend.lowering.JaxIrContext(*\*args*, *\*\*kwargs*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/interpreters/mlir.py#L615-L622)[\#](#jax.extend.lowering.JaxIrContext "Link to this definition")  
\_\_init\_\_(*self*) → [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)")[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/interpreters/mlir.py#L616-L622)[\#](#jax.extend.lowering.JaxIrContext.__init__ "Link to this definition")  
Creates a new MLIR context.

The context is the top-level container for all MLIR objects. It owns the storage for types, attributes, locations, and other core IR objects. A context can be configured to allow or disallow unregistered dialects and can have dialects loaded on-demand.

Methods

|  |  |
|----|----|
| [`__init__`](#jax.extend.lowering.JaxIrContext.__init__ "jax.extend.lowering.JaxIrContext.__init__")(self) | Creates a new MLIR context. |

Attributes

|  |  |
|----|----|
| `allow_unregistered_dialects` | Controls whether unregistered dialects are allowed in this context. |
| `append_dialect_registry` | Appends the contents of a dialect registry to the context. |
| `attach_diagnostic_handler` | Attaches a diagnostic handler that will receive callbacks. |
| `current` |  |
| `d` | Alias for dialects. |
| `dialects` | Gets a container for accessing dialects by name. |
| `emit_error_diagnostics` | Controls whether error diagnostics are emitted to diagnostic handlers. |
| `enable_multithreading` | Enables or disables multi-threading support in the context. |
| `get_dialect_descriptor` | Gets or loads a dialect by name, returning its descriptor object. |
| `get_num_threads` | Gets the number of threads in the context's thread pool. |
| `is_registered_operation` | Checks whether an operation with the given name is registered. |
| `load_all_available_dialects` | Loads all dialects available in the registry into the context. |
| `set_thread_pool` | Sets a custom thread pool for the context to use. |

[](../jax.extend.lowering.html "previous page")

previous

`jax.extend.lowering` module

[](jax.extend.lowering.LoweringRuleContext.html "next page")

next

jax.extend.lowering.LoweringRuleContext

Contents

- [`JaxIrContext`](#jax.extend.lowering.JaxIrContext)
  - [`JaxIrContext.__init__()`](#jax.extend.lowering.JaxIrContext.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
