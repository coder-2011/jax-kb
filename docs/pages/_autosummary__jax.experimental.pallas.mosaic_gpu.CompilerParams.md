- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.mosaic_gpu` module](../jax.experimental.pallas.mosaic_gpu.html)
- jax.experimental.pallas.mosaic_gpu.CompilerParams

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.mosaic_gpu.CompilerParams.rst "Download source file")
-  .pdf

# jax.experimental.pallas.mosaic_gpu.CompilerParams

## Contents

- [`CompilerParams`](#jax.experimental.pallas.mosaic_gpu.CompilerParams)
  - [`CompilerParams.approx_math`](#jax.experimental.pallas.mosaic_gpu.CompilerParams.approx_math)
  - [`CompilerParams.dimension_semantics`](#jax.experimental.pallas.mosaic_gpu.CompilerParams.dimension_semantics)
  - [`CompilerParams.max_concurrent_steps`](#jax.experimental.pallas.mosaic_gpu.CompilerParams.max_concurrent_steps)
  - [`CompilerParams.delay_release`](#jax.experimental.pallas.mosaic_gpu.CompilerParams.delay_release)
  - [`CompilerParams.unsafe_no_auto_barriers`](#jax.experimental.pallas.mosaic_gpu.CompilerParams.unsafe_no_auto_barriers)
  - [`CompilerParams.reduction_scratch_bytes`](#jax.experimental.pallas.mosaic_gpu.CompilerParams.reduction_scratch_bytes)
  - [`CompilerParams.profile_space`](#jax.experimental.pallas.mosaic_gpu.CompilerParams.profile_space)
  - [`CompilerParams.profile_dir`](#jax.experimental.pallas.mosaic_gpu.CompilerParams.profile_dir)
  - [`CompilerParams.profile_trace_scope`](#jax.experimental.pallas.mosaic_gpu.CompilerParams.profile_trace_scope)
  - [`CompilerParams.__init__()`](#jax.experimental.pallas.mosaic_gpu.CompilerParams.__init__)

# jax.experimental.pallas.mosaic_gpu.CompilerParams[\#](#jax-experimental-pallas-mosaic-gpu-compilerparams "Link to this heading")

*class* jax.experimental.pallas.mosaic_gpu.CompilerParams(*\**, *approx_math=False*, *dimension_semantics=None*, *max_concurrent_steps=1*, *unsafe_no_auto_barriers=False*, *reduction_scratch_bytes=2048*, *profile_space=0*, *profile_dir=''*, *profile_trace_scope=TraceScope.WARPGROUP*, *lowering_semantics=LoweringSemantics.Lane*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic_gpu/core.py#L85-L144)[\#](#jax.experimental.pallas.mosaic_gpu.CompilerParams "Link to this definition")  
Mosaic GPU compiler parameters.

Parameters:  
- **approx_math** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

- **dimension_semantics** (*Sequence\[DimensionSemantics\]* *\|* *None*)

- **max_concurrent_steps** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **unsafe_no_auto_barriers** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

- **reduction_scratch_bytes** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **profile_space** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **profile_dir** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)"))

- **profile_trace_scope** (*TraceScope*)

- **lowering_semantics** (*mgpu.core.LoweringSemantics*)

approx_math[\#](#jax.experimental.pallas.mosaic_gpu.CompilerParams.approx_math "Link to this definition")  
If True, the compiler is allowed to use approximate implementations of some math operations, e.g. `exp`. Defaults to False.

Type:  
[bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")

dimension_semantics[\#](#jax.experimental.pallas.mosaic_gpu.CompilerParams.dimension_semantics "Link to this definition")  
A list of dimension semantics for each grid dimension of the kernel. Either “parallel” for dimensions that can execute in any order, or “sequential” for dimensions that must be executed sequentially.

Type:  
Sequence\[DimensionSemantics\] \| None

max_concurrent_steps[\#](#jax.experimental.pallas.mosaic_gpu.CompilerParams.max_concurrent_steps "Link to this definition")  
The maximum number of sequential stages that are active concurrently. Defaults to 1.

Type:  
[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

delay_release[\#](#jax.experimental.pallas.mosaic_gpu.CompilerParams.delay_release "Link to this definition")  
The number of steps to wait before reusing the input/output references. Defaults to 0, and must be strictly smaller than max_concurrent_steps. Generally, you’ll want to set it to 1 if you don’t await the WGMMA in the body.

unsafe_no_auto_barriers[\#](#jax.experimental.pallas.mosaic_gpu.CompilerParams.unsafe_no_auto_barriers "Link to this definition")  
If True, Pallas will never automatically insert barrier instructions that ensure synchronous semantics of loads and stores. At the moment, the insertion is done conservatively and might regress performance. There are (at least) two conditions that must be satisfied for the use of this flag to be safe. First, no memory region is ever read *and* written to by the same thread (async copies are performed by background threads and do not count towards this rule). Secondly, no thread ever calls commit_smem(), reads from the committed SMEM and then issues an async copy overwriting that region (this is a very artificial and highly unlikely scenario).

Type:  
[bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")

reduction_scratch_bytes[\#](#jax.experimental.pallas.mosaic_gpu.CompilerParams.reduction_scratch_bytes "Link to this definition")  
The number of shared memory bytes to reserve as scratch space for cross-warp reductions. The higher this value, the more registers can be reduced in parallel. 2 \* 128 \* 6 \* 4 = 6144 bytes is typically a good value in order to extract most of the potential gains on H100 and B200.

Type:  
[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

profile_space[\#](#jax.experimental.pallas.mosaic_gpu.CompilerParams.profile_space "Link to this definition")  
The number of profiler events that can be collected in a single invocation. It is undefined behavior if a thread collects more events than this.

Type:  
[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

profile_dir[\#](#jax.experimental.pallas.mosaic_gpu.CompilerParams.profile_dir "Link to this definition")  
The directory to which profiling traces will be written to.

Type:  
[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")

profile_trace_scope[\#](#jax.experimental.pallas.mosaic_gpu.CompilerParams.profile_trace_scope "Link to this definition")  
The scope at which traces are collected (WARP or WARPGROUP).

Type:  
TraceScope

\_\_init\_\_(*\**, *approx_math=False*, *dimension_semantics=None*, *max_concurrent_steps=1*, *unsafe_no_auto_barriers=False*, *reduction_scratch_bytes=2048*, *profile_space=0*, *profile_dir=''*, *profile_trace_scope=TraceScope.WARPGROUP*, *lowering_semantics=LoweringSemantics.Lane*)[\#](#jax.experimental.pallas.mosaic_gpu.CompilerParams.__init__ "Link to this definition")  
Parameters:  
- **approx_math** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

- **dimension_semantics** (*Sequence\[DimensionSemantics\]* *\|* *None*)

- **max_concurrent_steps** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **unsafe_no_auto_barriers** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

- **reduction_scratch_bytes** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **profile_space** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **profile_dir** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)"))

- **profile_trace_scope** (*TraceScope*)

- **lowering_semantics** (*mgpu.core.LoweringSemantics*)

Return type:  
None

Methods

|  |  |
|----|----|
| [`__init__`](#jax.experimental.pallas.mosaic_gpu.CompilerParams.__init__ "jax.experimental.pallas.mosaic_gpu.CompilerParams.__init__")(\*\[, approx_math, ...\]) |  |
| `replace`(\*\*changes) | Return a new object replacing specified fields with new values. |

Attributes

|  |  |
|----|----|
| [`approx_math`](#jax.experimental.pallas.mosaic_gpu.CompilerParams.approx_math "jax.experimental.pallas.mosaic_gpu.CompilerParams.approx_math") |  |
| [`dimension_semantics`](#jax.experimental.pallas.mosaic_gpu.CompilerParams.dimension_semantics "jax.experimental.pallas.mosaic_gpu.CompilerParams.dimension_semantics") |  |
| `lowering_semantics` |  |
| [`max_concurrent_steps`](#jax.experimental.pallas.mosaic_gpu.CompilerParams.max_concurrent_steps "jax.experimental.pallas.mosaic_gpu.CompilerParams.max_concurrent_steps") |  |
| [`profile_dir`](#jax.experimental.pallas.mosaic_gpu.CompilerParams.profile_dir "jax.experimental.pallas.mosaic_gpu.CompilerParams.profile_dir") |  |
| [`profile_space`](#jax.experimental.pallas.mosaic_gpu.CompilerParams.profile_space "jax.experimental.pallas.mosaic_gpu.CompilerParams.profile_space") |  |
| [`profile_trace_scope`](#jax.experimental.pallas.mosaic_gpu.CompilerParams.profile_trace_scope "jax.experimental.pallas.mosaic_gpu.CompilerParams.profile_trace_scope") |  |
| [`reduction_scratch_bytes`](#jax.experimental.pallas.mosaic_gpu.CompilerParams.reduction_scratch_bytes "jax.experimental.pallas.mosaic_gpu.CompilerParams.reduction_scratch_bytes") |  |
| [`unsafe_no_auto_barriers`](#jax.experimental.pallas.mosaic_gpu.CompilerParams.unsafe_no_auto_barriers "jax.experimental.pallas.mosaic_gpu.CompilerParams.unsafe_no_auto_barriers") |  |

[](jax.experimental.pallas.mosaic_gpu.BlockSpec.html "previous page")

previous

jax.experimental.pallas.mosaic_gpu.BlockSpec

[](jax.experimental.pallas.mosaic_gpu.MemorySpace.html "next page")

next

jax.experimental.pallas.mosaic_gpu.MemorySpace

Contents

- [`CompilerParams`](#jax.experimental.pallas.mosaic_gpu.CompilerParams)
  - [`CompilerParams.approx_math`](#jax.experimental.pallas.mosaic_gpu.CompilerParams.approx_math)
  - [`CompilerParams.dimension_semantics`](#jax.experimental.pallas.mosaic_gpu.CompilerParams.dimension_semantics)
  - [`CompilerParams.max_concurrent_steps`](#jax.experimental.pallas.mosaic_gpu.CompilerParams.max_concurrent_steps)
  - [`CompilerParams.delay_release`](#jax.experimental.pallas.mosaic_gpu.CompilerParams.delay_release)
  - [`CompilerParams.unsafe_no_auto_barriers`](#jax.experimental.pallas.mosaic_gpu.CompilerParams.unsafe_no_auto_barriers)
  - [`CompilerParams.reduction_scratch_bytes`](#jax.experimental.pallas.mosaic_gpu.CompilerParams.reduction_scratch_bytes)
  - [`CompilerParams.profile_space`](#jax.experimental.pallas.mosaic_gpu.CompilerParams.profile_space)
  - [`CompilerParams.profile_dir`](#jax.experimental.pallas.mosaic_gpu.CompilerParams.profile_dir)
  - [`CompilerParams.profile_trace_scope`](#jax.experimental.pallas.mosaic_gpu.CompilerParams.profile_trace_scope)
  - [`CompilerParams.__init__()`](#jax.experimental.pallas.mosaic_gpu.CompilerParams.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
