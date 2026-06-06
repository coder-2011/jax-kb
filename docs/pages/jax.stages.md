- [](index.html)
- [API Reference](jax.html)
- `jax.stages` module

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](_sources/jax.stages.rst "Download source file")
-  .pdf

# jax.stages module

## Contents

- [Classes](#classes)
  - [`Wrapped`](#jax.stages.Wrapped)
    - [`Wrapped.__call__()`](#jax.stages.Wrapped.__call__)
    - [`Wrapped.lower()`](#jax.stages.Wrapped.lower)
    - [`Wrapped.trace()`](#jax.stages.Wrapped.trace)
  - [`Traced`](#jax.stages.Traced)
    - [`Traced.lower()`](#jax.stages.Traced.lower)
  - [`Lowered`](#jax.stages.Lowered)
    - [`Lowered.as_text()`](#jax.stages.Lowered.as_text)
    - [`Lowered.compile()`](#jax.stages.Lowered.compile)
    - [`Lowered.compiler_ir()`](#jax.stages.Lowered.compiler_ir)
    - [`Lowered.cost_analysis()`](#jax.stages.Lowered.cost_analysis)
    - [`Lowered.in_tree`](#jax.stages.Lowered.in_tree)
  - [`Compiled`](#jax.stages.Compiled)
    - [`Compiled.__call__()`](#jax.stages.Compiled.__call__)
    - [`Compiled.as_text()`](#jax.stages.Compiled.as_text)
    - [`Compiled.cost_analysis()`](#jax.stages.Compiled.cost_analysis)
    - [`Compiled.in_tree`](#jax.stages.Compiled.in_tree)
    - [`Compiled.memory_analysis()`](#jax.stages.Compiled.memory_analysis)
    - [`Compiled.runtime_executable()`](#jax.stages.Compiled.runtime_executable)

# `jax.stages` module[\#](#module-jax.stages "Link to this heading")

Interfaces to stages of the compiled execution process.

JAX transformations that compile just in time for execution, such as `jax.jit` and `jax.pmap`, also support a common means of explicit lowering and compilation *ahead of time*. This module defines types that represent the stages of this process.

For more, see the [AOT walkthrough](https://docs.jax.dev/en/latest/aot.html).

## Classes[\#](#classes "Link to this heading")

*class* jax.stages.Wrapped(*\*args*, *\*\*kwargs*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/stages.py#L912-L950)[\#](#jax.stages.Wrapped "Link to this definition")  
A function ready to be traced, lowered, and compiled.

This protocol reflects the output of functions such as `jax.jit`. Calling it results in JIT (just-in-time) lowering, compilation, and execution. It can also be explicitly lowered prior to compilation, and the result compiled prior to execution.

\_\_call\_\_(*\*args*, *\*\*kwargs*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/stages.py#L922-L925)[\#](#jax.stages.Wrapped.__call__ "Link to this definition")  
Executes the wrapped function, lowering and compiling as needed.

lower(*\*args*, *\*\*kwargs*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/stages.py#L937-L950)[\#](#jax.stages.Wrapped.lower "Link to this definition")  
Lower this function explicitly for the given arguments.

This is a shortcut for `self.trace(*args,`` ``**kwargs).lower()`.

A lowered function is staged out of Python and translated to a compiler’s input language, possibly in a backend-dependent manner. It is ready for compilation but not yet compiled.

Returns:  
A `Lowered` instance representing the lowering.

Return type:  
[Lowered](#jax.stages.Lowered "jax.stages.Lowered")

trace(*\*args*, *\*\*kwargs*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/stages.py#L926-L936)[\#](#jax.stages.Wrapped.trace "Link to this definition")  
Trace this function explicitly for the given arguments.

A traced function is staged out of Python and translated to a jaxpr. It is ready for lowering but not yet lowered.

Returns:  
A `Traced` instance representing the tracing.

Return type:  
[Traced](#jax.stages.Traced "jax.stages.Traced")

&nbsp;

*class* jax.stages.Traced(*meta_tys_flat*, *params*, *in_tree*, *out_tree*, *consts*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/stages.py#L403-L503)[\#](#jax.stages.Traced "Link to this definition")  
Traced form of a function specialized to argument types and values.

A traced computation is ready for lowering. This class carries the traced representation with the remaining information needed to later lower, compile, and execute it.

Provides access to both the hijax (high-level) and lojax (low-level) representations via .jaxpr and .lojax properties respectively.

lower(*\**, *lowering_platforms=None*, *\_private_parameters=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/stages.py#L480-L503)[\#](#jax.stages.Traced.lower "Link to this definition")  
Lower to compiler input, returning a `Lowered` instance.

Parameters:  
- **lowering_platforms** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*,* *...\]* *\|* *None*)

- **\_private_parameters** (*mlir.LoweringParameters* *\|* *None*)

&nbsp;

*class* jax.stages.Lowered(*lowering*, *args_info*, *out_tree*, *no_kwargs=False*, *in_types=None*, *out_types=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/stages.py#L531-L649)[\#](#jax.stages.Lowered "Link to this definition")  
Lowering of a function specialized to argument types and values.

A lowering is a computation ready for compilation. This class carries a lowering together with the remaining information needed to later compile and execute it. It also provides a common API for querying properties of lowered computations across JAX’s various lowering paths ([`jit()`](_autosummary/jax.jit.html#jax.jit "jax.jit"), [`pmap()`](_autosummary/jax.pmap.html#jax.pmap "jax.pmap"), etc.).

Parameters:  
- **lowering** (*Lowering*)

- **args_info** (*Any*)

- **out_tree** (*tree_util.PyTreeDef*)

- **no_kwargs** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

as_text(*dialect=None*, *\**, *debug_info=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/stages.py#L596-L611)[\#](#jax.stages.Lowered.as_text "Link to this definition")  
A human-readable text representation of this lowering.

Intended for visualization and debugging purposes. This need not be a valid nor reliable serialization. Use jax.export if you want reliable and portable serialization.

Parameters:  
- **dialect** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – Optional string specifying a lowering dialect (e.g. “stablehlo”, or “hlo”).

- **debug_info** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – Whether to include debugging information, e.g., source location.

Return type:  
[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")

compile(*compiler_options=None*, *\**, *device_assignment=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/stages.py#L586-L595)[\#](#jax.stages.Lowered.compile "Link to this definition")  
Compile, returning a corresponding `Compiled` instance.

Parameters:  
- **compiler_options** (*CompilerOptions* *\|* *None*)

- **device_assignment** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[xc.Device,* *...\]* *\|* *None*)

Return type:  
[Compiled](#jax.stages.Compiled "jax.stages.Compiled")

compiler_ir(*dialect=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/stages.py#L612-L631)[\#](#jax.stages.Lowered.compiler_ir "Link to this definition")  
An arbitrary object representation of this lowering.

Intended for debugging purposes. This is not a valid nor reliable serialization. The output has no guarantee of consistency across invocations. Use jax.export if you want reliable and portable serialization.

Returns `None` if unavailable, e.g. based on backend, compiler, or runtime.

Parameters:  
**dialect** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – Optional string specifying a lowering dialect (e.g. “stablehlo”, or “hlo”).

Return type:  
Any \| None

cost_analysis()[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/stages.py#L632-L649)[\#](#jax.stages.Lowered.cost_analysis "Link to this definition")  
A summary of execution cost estimates.

Intended for visualization and debugging purposes. The object output by this is some simple data structure that can easily be printed or serialized (e.g. nested dicts, lists, and tuples with numeric leaves). However, its structure can be arbitrary: it may be inconsistent across versions of JAX and jaxlib, or even across invocations.

Returns `None` if unavailable, e.g. based on backend, compiler, or runtime.

Return type:  
Any \| None

*property* in_tree*: tree_util.PyTreeDef*[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/stages.py#L337-L341)[\#](#jax.stages.Lowered.in_tree "Link to this definition")  
Tree structure of the pair (positional arguments, keyword arguments).

&nbsp;

*class* jax.stages.Compiled(*executable*, *const_args*, *args_info*, *out_tree*, *no_kwargs=False*, *in_types=None*, *out_types=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/stages.py#L651-L891)[\#](#jax.stages.Compiled "Link to this definition")  
Compiled representation of a function specialized to types/values.

A compiled computation is associated with an executable and the remaining information needed to execute it. It also provides a common API for querying properties of compiled computations across JAX’s various compilation paths and backends.

Parameters:  
- **const_args** ([*list*](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")*\[ArrayLike\]*)

- **args_info** (*Any*)

- **out_tree** (*tree_util.PyTreeDef*)

\_\_call\_\_(*\*args*, *\*\*kwargs*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/stages.py#L881-L891)[\#](#jax.stages.Compiled.__call__ "Link to this definition")  
Call self as a function.

as_text()[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/stages.py#L679-L692)[\#](#jax.stages.Compiled.as_text "Link to this definition")  
A human-readable text representation of this executable.

Intended for visualization and debugging purposes. This is not a valid nor reliable serialization.

Returns `None` if unavailable, e.g. based on backend, compiler, or runtime.

Return type:  
[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") \| None

cost_analysis()[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/stages.py#L693-L710)[\#](#jax.stages.Compiled.cost_analysis "Link to this definition")  
A summary of execution cost estimates.

Intended for visualization and debugging purposes. The object output by this is some simple data structure that can easily be printed or serialized (e.g. nested dicts, lists, and tuples with numeric leaves). However, its structure can be arbitrary: it may be inconsistent across versions of JAX and jaxlib, or even across invocations.

Returns `None` if unavailable, e.g. based on backend, compiler, or runtime.

Return type:  
Any \| None

*property* in_tree*: tree_util.PyTreeDef*[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/stages.py#L337-L341)[\#](#jax.stages.Compiled.in_tree "Link to this definition")  
Tree structure of the pair (positional arguments, keyword arguments).

memory_analysis()[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/stages.py#L711-L728)[\#](#jax.stages.Compiled.memory_analysis "Link to this definition")  
A summary of estimated memory requirements.

Intended for visualization and debugging purposes. The object output by this is some simple data structure that can easily be printed or serialized (e.g. nested dicts, lists, and tuples with numeric leaves). However, its structure can be arbitrary: it may be inconsistent across versions of JAX and jaxlib, or even across invocations.

Returns `None` if unavailable, e.g. based on backend, compiler, or runtime.

Return type:  
Any \| None

runtime_executable()[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/stages.py#L750-L761)[\#](#jax.stages.Compiled.runtime_executable "Link to this definition")  
An arbitrary object representation of this executable.

Intended for debugging purposes. This is not valid nor reliable serialization. The output has no guarantee of consistency across invocations.

Returns `None` if unavailable, e.g. based on backend, compiler, or runtime.

Return type:  
Any \| None

[](_autosummary/jax.ref.addupdate.html "previous page")

previous

jax.ref.addupdate

[](jax.test_util.html "next page")

next

`jax.test_util` module

Contents

- [Classes](#classes)
  - [`Wrapped`](#jax.stages.Wrapped)
    - [`Wrapped.__call__()`](#jax.stages.Wrapped.__call__)
    - [`Wrapped.lower()`](#jax.stages.Wrapped.lower)
    - [`Wrapped.trace()`](#jax.stages.Wrapped.trace)
  - [`Traced`](#jax.stages.Traced)
    - [`Traced.lower()`](#jax.stages.Traced.lower)
  - [`Lowered`](#jax.stages.Lowered)
    - [`Lowered.as_text()`](#jax.stages.Lowered.as_text)
    - [`Lowered.compile()`](#jax.stages.Lowered.compile)
    - [`Lowered.compiler_ir()`](#jax.stages.Lowered.compiler_ir)
    - [`Lowered.cost_analysis()`](#jax.stages.Lowered.cost_analysis)
    - [`Lowered.in_tree`](#jax.stages.Lowered.in_tree)
  - [`Compiled`](#jax.stages.Compiled)
    - [`Compiled.__call__()`](#jax.stages.Compiled.__call__)
    - [`Compiled.as_text()`](#jax.stages.Compiled.as_text)
    - [`Compiled.cost_analysis()`](#jax.stages.Compiled.cost_analysis)
    - [`Compiled.in_tree`](#jax.stages.Compiled.in_tree)
    - [`Compiled.memory_analysis()`](#jax.stages.Compiled.memory_analysis)
    - [`Compiled.runtime_executable()`](#jax.stages.Compiled.runtime_executable)

By The JAX authors

© Copyright 2024, The JAX Authors.\
