- [](../index.html)
- [Pallas: a JAX kernel language](index.html)
- Pallas Changelog

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .md](../_sources/pallas/CHANGELOG.md "Download source file")
-  .pdf

# Pallas Changelog

## Contents

- [Unreleased](#unreleased)
  - [TPU](#tpu)
  - [Mosaic GPU](#mosaic-gpu)
- [Released with JAX 0.10.1](#released-with-jax-0-10-1)
  - [TPU](#id2)
  - [Mosaic GPU](#id3)
- [Released with JAX 0.10.0](#released-with-jax-0-10-0)
  - [TPU](#id4)
  - [Mosaic GPU](#id5)
- [Released with JAX 0.9.2](#released-with-jax-0-9-2)
- [Released with JAX 0.9.1](#released-with-jax-0-9-1)
- [Released with JAX 0.9.0](#released-with-jax-0-9-0)
- [Released with jax 0.8.1](#released-with-jax-0-8-1)
- [Released with jax 0.7.1](#released-with-jax-0-7-1)
- [Released with jax 0.7.0](#released-with-jax-0-7-0)
- [Released with jax 0.6.1](#released-with-jax-0-6-1)
- [Released with jax 0.5.0](#released-with-jax-0-5-0)
- [Released with jax 0.4.37](#released-with-jax-0-4-37)
- [Released with jax 0.4.36 (December 6, 2024)](#released-with-jax-0-4-36-december-6-2024)
- [Released with jax 0.4.35 (October 22, 2024)](#released-with-jax-0-4-35-october-22-2024)
- [Released with jax 0.4.34 (October 4, 2024)](#released-with-jax-0-4-34-october-4-2024)
- [Released with jax 0.4.33 (September 16, 2024)](#released-with-jax-0-4-33-september-16-2024)
- [Released with jax 0.4.32 (September 11, 2024)](#released-with-jax-0-4-32-september-11-2024)
- [Released with jax 0.4.31 (July 29, 2024)](#released-with-jax-0-4-31-july-29-2024)
- [Released with JAX 0.4.30 (June 18, 2024)](#released-with-jax-0-4-30-june-18-2024)

# Pallas Changelog[\#](#pallas-changelog "Link to this heading")

This is the list of changes specific to [`jax.experimental.pallas`](../jax.experimental.pallas.html#module-jax.experimental.pallas "jax.experimental.pallas"). For the overall JAX change log see [here](https://docs.jax.dev/en/latest/changelog.html).

## Unreleased[\#](#unreleased "Link to this heading")

- New features

  - Added `jax.experimental.pallas.enable_poison_buffers` (config flag `jax_pallas_poison_buffers`) to poison (initialize with NaNs or lowest possible integers) any scratch buffers allocated by Pallas in Mosaic TPU lowering for debugging.

  - Support TMA scatter.

  - Rename `TMA_GATHER_INDICES_LAYOUT` to `TMA_INDICES_LAYOUT`.

- Deprecations

  - `pl.debug_checks_enabled` is deprecated. Use `pl.enable_debug_checks.value`.

  - `pl.dot` was moved into [`jax.experimental.pallas.triton`](../jax.experimental.pallas.triton.html#module-jax.experimental.pallas.triton "jax.experimental.pallas.triton"). Accessing it via [`jax.experimental.pallas`](../jax.experimental.pallas.html#module-jax.experimental.pallas "jax.experimental.pallas") is deprecated. You can use [`jax.numpy.dot()`](../_autosummary/jax.numpy.dot.html#jax.numpy.dot "jax.numpy.dot"), [`jax.numpy.einsum()`](../_autosummary/jax.numpy.einsum.html#jax.numpy.einsum "jax.numpy.einsum") or the `@` operator instead in a TPU or MGPU kernel.

### TPU[\#](#tpu "Link to this heading")

- Changes

  - [`jax.experimental.pallas.tpu.CompilerParams`](../_autosummary/jax.experimental.pallas.tpu.CompilerParams.html#jax.experimental.pallas.tpu.CompilerParams "jax.experimental.pallas.tpu.CompilerParams") now defaults `needs_layout_passes` to True. The layout passes are still a work in progress. Please file a bug if you encounter a compilation error with them enabled.

  - [`jax.experimental.pallas.tpu.CompilerParams`](../_autosummary/jax.experimental.pallas.tpu.CompilerParams.html#jax.experimental.pallas.tpu.CompilerParams "jax.experimental.pallas.tpu.CompilerParams") now defaults `use_tc_tiling_on_sc` to True for SparseCore kernels.

- Removals

  - Removed the previously deprecated `jax.experimental.pallas.tpu.KernelType` and `jax.experimental.pallas.tpu.repeat()`.

- Deprecations

  - Deprecated `pltpu.HOST` and `pltpu.MemorySpace.HOST` in favor of `pl.HOST`.

### Mosaic GPU[\#](#mosaic-gpu "Link to this heading")

- New features

  - Support using [`jax.experimental.pallas.multiple_of()`](../_autosummary/jax.experimental.pallas.multiple_of.html#jax.experimental.pallas.multiple_of "jax.experimental.pallas.multiple_of") to specify divisibility requirements on dynamic indices.

  - Support allocating multidimensional `plgpu.Barrier`s and `plgpu.ClusterBarrier`s, by providing a nD shape as the `num_barriers` parameter.

- Changes

  - Breaking change: [`jax.experimental.pallas.program_id()`](../_autosummary/jax.experimental.pallas.program_id.html#jax.experimental.pallas.program_id "jax.experimental.pallas.program_id") and [`jax.experimental.pallas.num_programs()`](../_autosummary/jax.experimental.pallas.num_programs.html#jax.experimental.pallas.num_programs "jax.experimental.pallas.num_programs") no longer work inside kernels defined via [`jax.experimental.pallas.mosaic_gpu.kernel()`](../_autosummary/jax.experimental.pallas.mosaic_gpu.kernel.html#jax.experimental.pallas.mosaic_gpu.kernel "jax.experimental.pallas.mosaic_gpu.kernel"). Use [`jax.lax.axis_index()`](../_autosummary/jax.lax.axis_index.html#jax.lax.axis_index "jax.lax.axis_index") and [`jax.lax.axis_size()`](../_autosummary/jax.lax.axis_size.html#jax.lax.axis_size "jax.lax.axis_size") instead.

- Removals

  - Deleted `plgpu.unswizzle_ref` and `plgpu.untile_ref`.

- Deprecations

  - Using [`jax.experimental.pallas.pallas_call()`](../_autosummary/jax.experimental.pallas.pallas_call.html#jax.experimental.pallas.pallas_call "jax.experimental.pallas.pallas_call") for Mosaic GPU kernels is deprecated. Please migrate to [`jax.experimental.pallas.mosaic_gpu.kernel()`](../_autosummary/jax.experimental.pallas.mosaic_gpu.kernel.html#jax.experimental.pallas.mosaic_gpu.kernel "jax.experimental.pallas.mosaic_gpu.kernel") and [`jax.experimental.pallas.mosaic_gpu.emit_pipeline()`](../_autosummary/jax.experimental.pallas.mosaic_gpu.emit_pipeline.html#jax.experimental.pallas.mosaic_gpu.emit_pipeline "jax.experimental.pallas.mosaic_gpu.emit_pipeline").

## Released with JAX 0.10.1[\#](#released-with-jax-0-10-1 "Link to this heading")

- Changes

  - Added `jax.experimental.pallas.align_to()`, a utility that rounds a value up to the nearest multiple of a given alignment.

  - [`jax.experimental.pallas.pallas_call()`](../_autosummary/jax.experimental.pallas.pallas_call.html#jax.experimental.pallas.pallas_call "jax.experimental.pallas.pallas_call") no longer supports checkify. We expect this change to affect few users, as in our experience most kernels either perform no checking or use [`jax.experimental.pallas.debug_check()`](../_autosummary/jax.experimental.pallas.debug_check.html#jax.experimental.pallas.debug_check "jax.experimental.pallas.debug_check") for conditionally-enabled runtime checks.

  - [`jax.experimental.pallas.kernel()`](../_autosummary/jax.experimental.pallas.kernel.html#jax.experimental.pallas.kernel "jax.experimental.pallas.kernel") now always aliases Refs that are passed in or closed-over.

### TPU[\#](#id2 "Link to this heading")

- Removals

  - Removed the `kernel_type` field from [`jax.experimental.pallas.tpu.CompilerParams`](../_autosummary/jax.experimental.pallas.tpu.CompilerParams.html#jax.experimental.pallas.tpu.CompilerParams "jax.experimental.pallas.tpu.CompilerParams"). It was only used for writing SparseCore kernels via [`jax.experimental.pallas.pallas_call()`](../_autosummary/jax.experimental.pallas.pallas_call.html#jax.experimental.pallas.pallas_call "jax.experimental.pallas.pallas_call"), which is now unsupported. The recommended API for SparseCore kernels is [`jax.experimental.pallas.kernel()`](../_autosummary/jax.experimental.pallas.kernel.html#jax.experimental.pallas.kernel "jax.experimental.pallas.kernel").

### Mosaic GPU[\#](#id3 "Link to this heading")

- New features

  - Added `jax.experimental.pallas.mosaic_gpu.barrier_test()` function; a non-blocking equivalent of [`jax.experimental.pallas.mosaic_gpu.barrier_wait()`](../_autosummary/jax.experimental.pallas.mosaic_gpu.barrier_wait.html#jax.experimental.pallas.mosaic_gpu.barrier_wait "jax.experimental.pallas.mosaic_gpu.barrier_wait") only supported in a warp context.

- Changes

  - Breaking change: removed `plgpu.TransposeTransform`.

## Released with JAX 0.10.0[\#](#released-with-jax-0-10-0 "Link to this heading")

- Changes

  - Breaking change: refactored `pl.kernel` to use `out_type` instead of `out_shape` and `scratch_types` instead of `scratch_shapes`. Existing usages calling [`jax.experimental.pallas.kernel()`](../_autosummary/jax.experimental.pallas.kernel.html#jax.experimental.pallas.kernel "jax.experimental.pallas.kernel") with these keyword arguments must be updated.

### TPU[\#](#id4 "Link to this heading")

- Deprecations

  - `pltpu.semaphore`, `pltpu.DeviceIdType`, `pltpu.semaphore_signal`, `pltpu.semaphore_wait`, and `pltpu.semaphore_read` are now available in [`jax.experimental.pallas`](../jax.experimental.pallas.html#module-jax.experimental.pallas "jax.experimental.pallas"). Accessing them via [`jax.experimental.pallas.tpu`](../jax.experimental.pallas.tpu.html#module-jax.experimental.pallas.tpu "jax.experimental.pallas.tpu") is deprecated.

- Removals

  - Removed the previously deprecated `pltpu.ANY` and `pltpu.MemorySpace.ANY`. Use `pl.ANY` instead.

  - Removed the deprecated `pltpu.delay`, which is now available as `pl.delay`.

### Mosaic GPU[\#](#id5 "Link to this heading")

- New features

  - Added the `leader_tracked` argument to `ClusterBarrier`, which allows tracking barrier completions solely from the leader block along a specific axis in a cluster.

## Released with JAX 0.9.2[\#](#released-with-jax-0-9-2 "Link to this heading")

- New features:

  - Added support for atomics to the Mosaic GPU backend, through `jax.experimental.pallas.mosaic_gpu.atomic_add()`, `jax.experimental.pallas.mosaic_gpu.atomic_max()`, `jax.experimental.pallas.mosaic_gpu.atomic_min()`, `jax.experimental.pallas.mosaic_gpu.atomic_and()`, `jax.experimental.pallas.mosaic_gpu.atomic_or()`, and `jax.experimental.pallas.mosaic_gpu.atomic_xor()`.

  - Added a `leader_tracked` argument to [`jax.experimental.pallas.mosaic_gpu.copy_gmem_to_smem()`](../_autosummary/jax.experimental.pallas.mosaic_gpu.copy_gmem_to_smem.html#jax.experimental.pallas.mosaic_gpu.copy_gmem_to_smem "jax.experimental.pallas.mosaic_gpu.copy_gmem_to_smem"). This adds support for cluster-collective partitioned/replicated GMEM to SMEM copies whose completion is tracked only by the leader block. Removed the `partitioned_axis` argument (which provided limited access to this feature) from the API.

## Released with JAX 0.9.1[\#](#released-with-jax-0-9-1 "Link to this heading")

- New features:

  - Added a [`jax.experimental.pallas.with_scoped()`](../_autosummary/jax.experimental.pallas.with_scoped.html#jax.experimental.pallas.with_scoped "jax.experimental.pallas.with_scoped") decorator that provides the function with scope-allocated scratch buffers.

- Changes

  - Removed the `backend` argument of [`jax.experimental.pallas.pallas_call()`](../_autosummary/jax.experimental.pallas.pallas_call.html#jax.experimental.pallas.pallas_call "jax.experimental.pallas.pallas_call") in favor of `compiler_params`. For example, to force the use of the Triton backend you have to now write `compiler_params=pltriton.CompilerParams()`, where `pltriton` refers to [`jax.experimental.pallas.triton`](../jax.experimental.pallas.triton.html#module-jax.experimental.pallas.triton "jax.experimental.pallas.triton").

  - Renamed `jax.experimental.pallas.tpu.KernelType` to `CoreType`. The old name is deprecated and will be removed in a future release.

## Released with JAX 0.9.0[\#](#released-with-jax-0-9-0 "Link to this heading")

- New features:

  - Added a `reduction_scratch_bytes` field to [`jax.experimental.pallas.mosaic_gpu.CompilerParams`](../_autosummary/jax.experimental.pallas.mosaic_gpu.CompilerParams.html#jax.experimental.pallas.mosaic_gpu.CompilerParams "jax.experimental.pallas.mosaic_gpu.CompilerParams"). This gives user control over how much shared memory Pallas is allowed to reserve for cross-warp reductions on GPU. Increasing this value typically allows for faster reductions.

- Changes

  - The default lowering path on GPU now goes through Mosaic GPU. To keep using Triton, call [`jax.experimental.pallas.pallas_call()`](../_autosummary/jax.experimental.pallas.pallas_call.html#jax.experimental.pallas.pallas_call "jax.experimental.pallas.pallas_call") with the `backend` argument set to `'triton'`.

- Removals

  - Removed the previously deprecated `pl.atomic_*`, `pl.load`, `pl.store`, `pl.swap` and `pl.max_contiguous`.

## Released with jax 0.8.1[\#](#released-with-jax-0-8-1 "Link to this heading")

- New features:

  - Added [`jax.experimental.pallas.tpu.get_tpu_info()`](../_autosummary/jax.experimental.pallas.tpu.get_tpu_info.html#jax.experimental.pallas.tpu.get_tpu_info "jax.experimental.pallas.tpu.get_tpu_info") to get TPU hardware information.

- Deprecations

  - `pl.max_contiguous` has been moved to [`jax.experimental.pallas.triton`](../jax.experimental.pallas.triton.html#module-jax.experimental.pallas.triton "jax.experimental.pallas.triton"). Accessing it via [`jax.experimental.pallas`](../jax.experimental.pallas.html#module-jax.experimental.pallas "jax.experimental.pallas") is deprecated.

  - `pl.swap` is deprecated and will be removed in a future release. Use indexing or backend-specific loading/storing APIs instead.

- Removals

  - Removed the previously deprecated `jax.experimental.pallas.tpu.TPUCompilerParams`, `jax.experimental.pallas.tpu.TPUMemorySpace`, `jax.experimental.pallas.tpu.TritonCompilerParams`.

## Released with jax 0.7.1[\#](#released-with-jax-0-7-1 "Link to this heading")

- New features:

  - `pltpu.make_async_remote_copy` and `pltpu.semaphore_signal`’s `device_id` argument now allows user to pass in a dictionary that only specifies the device index along the communication axis, instead of the full coordinates. It also supports TPU core id index.

  - `jax.debug.print` now works in Pallas kernels and is the recommended way to print.

- Deprecations

  - `pl.atomic_*` APIs have been moved to [`jax.experimental.pallas.triton`](../jax.experimental.pallas.triton.html#module-jax.experimental.pallas.triton "jax.experimental.pallas.triton"). Accessing them via [`jax.experimental.pallas`](../jax.experimental.pallas.html#module-jax.experimental.pallas "jax.experimental.pallas") is deprecated.

  - `pl.load` and `pl.store` are deprecated. Use indexing or backend-specific loading/storing APIs instead.

## Released with jax 0.7.0[\#](#released-with-jax-0-7-0 "Link to this heading")

- New functionality

  - Added a new decorator [`jax.experimental.pallas.loop()`](../_autosummary/jax.experimental.pallas.loop.html#jax.experimental.pallas.loop "jax.experimental.pallas.loop") which allows to write stateless loops as functions.

  - Added new multiple buffering and lookahead functionality to [`jax.experimental.pallas.tpu.emit_pipeline()`](../_autosummary/jax.experimental.pallas.tpu.emit_pipeline.html#jax.experimental.pallas.tpu.emit_pipeline "jax.experimental.pallas.tpu.emit_pipeline"). Input buffers can now be multiple-buffered with more than 2 buffers and support a lookahead option to fetch blocks that are an arbitrary number of grid iterations ahead rather than the immediate next iterations. Additionally, pipeline state can now be held in registers to reduce scalar memory usage.

- Deprecations

  - `jax.experimental.pallas.triton.TritonCompilerParams` has been renamed to [`jax.experimental.pallas.triton.CompilerParams`](../_autosummary/jax.experimental.pallas.triton.CompilerParams.html#jax.experimental.pallas.triton.CompilerParams "jax.experimental.pallas.triton.CompilerParams"). The old name is deprecated and will be removed in a future release.

  - `jax.experimental.pallas.tpu.TPUCompilerParams` and `jax.experimental.pallas.tpu.TPUMemorySpace` have been renamed to [`jax.experimental.pallas.tpu.CompilerParams`](../_autosummary/jax.experimental.pallas.tpu.CompilerParams.html#jax.experimental.pallas.tpu.CompilerParams "jax.experimental.pallas.tpu.CompilerParams") and [`jax.experimental.pallas.tpu.MemorySpace`](../_autosummary/jax.experimental.pallas.tpu.MemorySpace.html#jax.experimental.pallas.tpu.MemorySpace "jax.experimental.pallas.tpu.MemorySpace"). The old names are deprecated and will be removed in a future release.

## Released with jax 0.6.1[\#](#released-with-jax-0-6-1 "Link to this heading")

- Removals

  - Removed previously deprecated `jax.experimental.pallas.gpu`. To use the Triton backend import [`jax.experimental.pallas.triton`](../jax.experimental.pallas.triton.html#module-jax.experimental.pallas.triton "jax.experimental.pallas.triton").

- Changes

  - [`jax.experimental.pallas.BlockSpec()`](../_autosummary/jax.experimental.pallas.BlockSpec.html#jax.experimental.pallas.BlockSpec "jax.experimental.pallas.BlockSpec") now takes in special types in addition to ints/None in the `block_shape`. `indexing_mode` has been removed. To achieve “Unblocked”, pass a `pl.Element(size)` into `block_shape` for each entry that needs unblocked indexing.

  - [`jax.experimental.pallas.pallas_call()`](../_autosummary/jax.experimental.pallas.pallas_call.html#jax.experimental.pallas.pallas_call "jax.experimental.pallas.pallas_call") now requires `compiler_params` to be a backend-specific dataclass instead of a param to value mapping.

  - [`jax.experimental.pallas.debug_check()`](../_autosummary/jax.experimental.pallas.debug_check.html#jax.experimental.pallas.debug_check "jax.experimental.pallas.debug_check") is now supported both on TPU and Mosaic GPU. Previously, this functionality was only supported on TPU and required using the APIs from [`jax.experimental.checkify`](../jax.experimental.checkify.html#module-jax.experimental.checkify "jax.experimental.checkify"). Note that debug checks are not executed unless `jax.experimental.pallas.enable_debug_checks` is set.

## Released with jax 0.5.0[\#](#released-with-jax-0-5-0 "Link to this heading")

- New functionality

  - Added vector support for [`jax.experimental.pallas.debug_print()`](../_autosummary/jax.experimental.pallas.debug_print.html#jax.experimental.pallas.debug_print "jax.experimental.pallas.debug_print") on TPU.

## Released with jax 0.4.37[\#](#released-with-jax-0-4-37 "Link to this heading")

- New functionality

  - Added support for `DotAlgorithmPreset` precision arguments for `dot` lowering on Triton backend.

## Released with jax 0.4.36 (December 6, 2024)[\#](#released-with-jax-0-4-36-december-6-2024 "Link to this heading")

## Released with jax 0.4.35 (October 22, 2024)[\#](#released-with-jax-0-4-35-october-22-2024 "Link to this heading")

- Removals

  - Removed previously deprecated aliases `jax.experimental.pallas.tpu.CostEstimate` and `jax.experimental.tpu.run_scoped()`. Both are now available in [`jax.experimental.pallas`](../jax.experimental.pallas.html#module-jax.experimental.pallas "jax.experimental.pallas").

- New functionality

  - Added a cost estimate tool `pl.estimate_cost()` for automatically constructing a kernel cost estimate from a JAX reference function.

## Released with jax 0.4.34 (October 4, 2024)[\#](#released-with-jax-0-4-34-october-4-2024 "Link to this heading")

- Changes

  - [`jax.experimental.pallas.debug_print()`](../_autosummary/jax.experimental.pallas.debug_print.html#jax.experimental.pallas.debug_print "jax.experimental.pallas.debug_print") no longer requires all arguments to be scalars. The restrictions on the arguments are backend-specific: Non-scalar arguments are currently only supported on GPU, when using Triton.

  - [`jax.experimental.pallas.BlockSpec`](../_autosummary/jax.experimental.pallas.BlockSpec.html#jax.experimental.pallas.BlockSpec "jax.experimental.pallas.BlockSpec") no longer supports the previously deprecated argument order, where `index_map` comes before `block_shape`.

- Deprecations

  - The `jax.experimental.pallas.gpu` submodule is deprecated to avoid ambiguite with [`jax.experimental.pallas.mosaic_gpu`](../jax.experimental.pallas.mosaic_gpu.html#module-jax.experimental.pallas.mosaic_gpu "jax.experimental.pallas.mosaic_gpu"). To use the Triton backend import [`jax.experimental.pallas.triton`](../jax.experimental.pallas.triton.html#module-jax.experimental.pallas.triton "jax.experimental.pallas.triton").

- New functionality

  - [`jax.experimental.pallas.pallas_call()`](../_autosummary/jax.experimental.pallas.pallas_call.html#jax.experimental.pallas.pallas_call "jax.experimental.pallas.pallas_call") now accepts `scratch_shapes`, a PyTree specifying backend-specific temporary objects needed by the kernel, for example, buffers, synchronization primitives etc.

  - `checkify.check()` can now be used to insert runtime asserts when pallas_call is called with the `pltpu.enable_runtime_assert(True)` context manager.

## Released with jax 0.4.33 (September 16, 2024)[\#](#released-with-jax-0-4-33-september-16-2024 "Link to this heading")

## Released with jax 0.4.32 (September 11, 2024)[\#](#released-with-jax-0-4-32-september-11-2024 "Link to this heading")

- Changes

  - The kernel function is not allowed to close over constants. Instead, all the needed arrays must be passed as inputs, with proper block specs ([\#22746](https://github.com/jax-ml/jax/issues/22746)).

- New functionality

  - Improved error messages for mistakes in the signature of the index map functions, to include the name and source location of the index map.

## Released with jax 0.4.31 (July 29, 2024)[\#](#released-with-jax-0-4-31-july-29-2024 "Link to this heading")

- Changes

  - [`jax.experimental.pallas.BlockSpec`](../_autosummary/jax.experimental.pallas.BlockSpec.html#jax.experimental.pallas.BlockSpec "jax.experimental.pallas.BlockSpec") now expects `block_shape` to be passed *before* `index_map`. The old argument order is deprecated and will be removed in a future release.

  - [`jax.experimental.pallas.GridSpec`](../_autosummary/jax.experimental.pallas.GridSpec.html#jax.experimental.pallas.GridSpec "jax.experimental.pallas.GridSpec") does not have anymore the `in_specs_tree`, and the `out_specs_tree` fields, and the `in_specs` and `out_specs` tree now store the values as pytrees of BlockSpec. Previously, `in_specs` and `out_specs` were flattened ([\#22552](https://github.com/jax-ml/jax/issues/22552)).

  - The method `compute_index` of [`jax.experimental.pallas.GridSpec`](../_autosummary/jax.experimental.pallas.GridSpec.html#jax.experimental.pallas.GridSpec "jax.experimental.pallas.GridSpec") has been removed because it is private. Similarly, the `get_grid_mapping` and `unzip_dynamic_bounds` have been removed from `BlockSpec` ([\#22593](https://github.com/jax-ml/jax/issues/22593)).

  - Fixed the interpret mode to work with BlockSpec that involve padding ([\#22275](https://github.com/jax-ml/jax/issues/22275)). Padding in interpret mode will be with NaN, to help debug out-of-bounds errors, but this behavior is not present when running in custom kernel mode, and should not be depended on.

  - Previously it was possible to import many APIs that are meant to be private, as `jax.experimental.pallas.pallas`. This is not possible anymore.

- New Functionality

  - Added documentation for BlockSpec: [Grids and BlockSpecs](grid_blockspec.html#pallas-grids-and-blockspecs).

  - Improved error messages for the [`jax.experimental.pallas.pallas_call()`](../_autosummary/jax.experimental.pallas.pallas_call.html#jax.experimental.pallas.pallas_call "jax.experimental.pallas.pallas_call") API.

  - Added lowering rules for TPU for `lax.shift_right_arithmetic` ([\#22279](https://github.com/jax-ml/jax/issues/22279)) and `lax.erf_inv` ([\#22310](https://github.com/jax-ml/jax/issues/22310)).

  - Added initial support for shape polymorphism for the Pallas TPU custom kernels\
    ([\#22084](https://github.com/jax-ml/jax/issues/22084)).

  - Added TPU support for checkify. ([\#22480](https://github.com/jax-ml/jax/issues/22480))

  - Added clearer error messages when the block sizes do not match the TPU requirements. Previously, the errors were coming from the Mosaic backend and did not have useful Python stack traces.

  - Added support for TPU lowering with 1D blocks, and relaxed the requirements for the block sizes with at least 2 dimensions: the last 2 dimensions must be divisible by 8 and 128 respectively, unless they span the entire corresponding array dimension. Previously, block dimensions that spanned the entire array were allowed only if the block dimensions in the last two dimensions were smaller than 8 and 128 respectively.

## Released with JAX 0.4.30 (June 18, 2024)[\#](#released-with-jax-0-4-30-june-18-2024 "Link to this heading")

- New Functionality

  - Added checkify support for [`jax.experimental.pallas.pallas_call()`](../_autosummary/jax.experimental.pallas.pallas_call.html#jax.experimental.pallas.pallas_call "jax.experimental.pallas.pallas_call") in interpret mode ([\#21862](https://github.com/jax-ml/jax/issues/21862)).

  - Improved support for PRNG keys for TPU kernels ([\#21773](https://github.com/jax-ml/jax/issues/21773)).

[](design/async_note.html "previous page")

previous

Pallas Async Operations

[](../about.html "next page")

next

About the project

Contents

- [Unreleased](#unreleased)
  - [TPU](#tpu)
  - [Mosaic GPU](#mosaic-gpu)
- [Released with JAX 0.10.1](#released-with-jax-0-10-1)
  - [TPU](#id2)
  - [Mosaic GPU](#id3)
- [Released with JAX 0.10.0](#released-with-jax-0-10-0)
  - [TPU](#id4)
  - [Mosaic GPU](#id5)
- [Released with JAX 0.9.2](#released-with-jax-0-9-2)
- [Released with JAX 0.9.1](#released-with-jax-0-9-1)
- [Released with JAX 0.9.0](#released-with-jax-0-9-0)
- [Released with jax 0.8.1](#released-with-jax-0-8-1)
- [Released with jax 0.7.1](#released-with-jax-0-7-1)
- [Released with jax 0.7.0](#released-with-jax-0-7-0)
- [Released with jax 0.6.1](#released-with-jax-0-6-1)
- [Released with jax 0.5.0](#released-with-jax-0-5-0)
- [Released with jax 0.4.37](#released-with-jax-0-4-37)
- [Released with jax 0.4.36 (December 6, 2024)](#released-with-jax-0-4-36-december-6-2024)
- [Released with jax 0.4.35 (October 22, 2024)](#released-with-jax-0-4-35-october-22-2024)
- [Released with jax 0.4.34 (October 4, 2024)](#released-with-jax-0-4-34-october-4-2024)
- [Released with jax 0.4.33 (September 16, 2024)](#released-with-jax-0-4-33-september-16-2024)
- [Released with jax 0.4.32 (September 11, 2024)](#released-with-jax-0-4-32-september-11-2024)
- [Released with jax 0.4.31 (July 29, 2024)](#released-with-jax-0-4-31-july-29-2024)
- [Released with JAX 0.4.30 (June 18, 2024)](#released-with-jax-0-4-30-june-18-2024)

By The JAX authors

© Copyright 2024, The JAX Authors.\
