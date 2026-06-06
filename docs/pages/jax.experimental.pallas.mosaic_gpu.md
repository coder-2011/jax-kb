- [](index.html)
- [API Reference](jax.html)
- [`jax.experimental` module](jax.experimental.html)
- [`jax.experimental.pallas` module](jax.experimental.pallas.html)
- `jax.experimental.pallas.mosaic_gpu` module

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](_sources/jax.experimental.pallas.mosaic_gpu.rst "Download source file")
-  .pdf

# jax.experimental.pallas.mosaic_gpu module

## Contents

- [Classes](#classes)
- [Functions](#functions)
- [Loop-like functions](#loop-like-functions)
- [Synchronization](#synchronization)
- [Asynchronous copies](#asynchronous-copies)
- [Hopper-specific functions](#hopper-specific-functions)
- [Blackwell-specific functions](#blackwell-specific-functions)
- [Multimem operations](#multimem-operations)
- [Aliases](#aliases)

# `jax.experimental.pallas.mosaic_gpu` module[\#](#module-jax.experimental.pallas.mosaic_gpu "Link to this heading")

Experimental GPU backend for Pallas targeting H100.

These APIs are highly unstable and can change weekly. Use at your own risk.

## Classes[\#](#classes "Link to this heading")

|  |  |
|----|----|
| [`Barrier`](_autosummary/jax.experimental.pallas.mosaic_gpu.Barrier.html#jax.experimental.pallas.mosaic_gpu.Barrier "jax.experimental.pallas.mosaic_gpu.Barrier")(\*\[, num_arrivals, num_barriers, ...\]) | Describes a barrier reference. |
| [`BlockSpec`](_autosummary/jax.experimental.pallas.mosaic_gpu.BlockSpec.html#jax.experimental.pallas.mosaic_gpu.BlockSpec "jax.experimental.pallas.mosaic_gpu.BlockSpec")(\[block_shape, index_map, ...\]) | A GPU-specific `BlockSpec`. |
| [`CompilerParams`](_autosummary/jax.experimental.pallas.mosaic_gpu.CompilerParams.html#jax.experimental.pallas.mosaic_gpu.CompilerParams "jax.experimental.pallas.mosaic_gpu.CompilerParams")(\*\[, approx_math, ...\]) | Mosaic GPU compiler parameters. |
| [`MemorySpace`](_autosummary/jax.experimental.pallas.mosaic_gpu.MemorySpace.html#jax.experimental.pallas.mosaic_gpu.MemorySpace "jax.experimental.pallas.mosaic_gpu.MemorySpace")(value\[, names, module, ...\]) |  |
| [`Layout`](_autosummary/jax.experimental.pallas.mosaic_gpu.Layout.html#jax.experimental.pallas.mosaic_gpu.Layout "jax.experimental.pallas.mosaic_gpu.Layout")(value\[, names, module, qualname, ...\]) |  |
| [`SemaphoreType`](_autosummary/jax.experimental.pallas.mosaic_gpu.SemaphoreType.html#jax.experimental.pallas.mosaic_gpu.SemaphoreType "jax.experimental.pallas.mosaic_gpu.SemaphoreType")(value\[, names, module, ...\]) |  |
| [`SwizzleTransform`](_autosummary/jax.experimental.pallas.mosaic_gpu.SwizzleTransform.html#jax.experimental.pallas.mosaic_gpu.SwizzleTransform "jax.experimental.pallas.mosaic_gpu.SwizzleTransform")(swizzle) |  |
| [`TilingTransform`](_autosummary/jax.experimental.pallas.mosaic_gpu.TilingTransform.html#jax.experimental.pallas.mosaic_gpu.TilingTransform "jax.experimental.pallas.mosaic_gpu.TilingTransform")(tiling) | Represents a tiling transformation for memory refs. |
| [`WGMMAAccumulatorRef`](_autosummary/jax.experimental.pallas.mosaic_gpu.WGMMAAccumulatorRef.html#jax.experimental.pallas.mosaic_gpu.WGMMAAccumulatorRef "jax.experimental.pallas.mosaic_gpu.WGMMAAccumulatorRef")(shape, dtype, \_init) |  |

## Functions[\#](#functions "Link to this heading")

|  |  |
|----|----|
| [`as_torch_kernel`](_autosummary/jax.experimental.pallas.mosaic_gpu.as_torch_kernel.html#jax.experimental.pallas.mosaic_gpu.as_torch_kernel "jax.experimental.pallas.mosaic_gpu.as_torch_kernel")(fn) | Makes a Mosaic GPU kernel callable with PyTorch tensors. |
| [`kernel`](_autosummary/jax.experimental.pallas.mosaic_gpu.kernel.html#jax.experimental.pallas.mosaic_gpu.kernel "jax.experimental.pallas.mosaic_gpu.kernel")(body, out_shape, \*\[, scratch_shapes, ...\]) | Entry point for defining a Mosaic GPU kernel. |
| [`layout_cast`](_autosummary/jax.experimental.pallas.mosaic_gpu.layout_cast.html#jax.experimental.pallas.mosaic_gpu.layout_cast "jax.experimental.pallas.mosaic_gpu.layout_cast")(x, new_layout) | Casts the layout of the given array. |
| [`set_max_registers`](_autosummary/jax.experimental.pallas.mosaic_gpu.set_max_registers.html#jax.experimental.pallas.mosaic_gpu.set_max_registers "jax.experimental.pallas.mosaic_gpu.set_max_registers")(n, \*, action) | Sets the maximum number of per-lane registers in the thread. |
| [`planar_snake`](_autosummary/jax.experimental.pallas.mosaic_gpu.planar_snake.html#jax.experimental.pallas.mosaic_gpu.planar_snake "jax.experimental.pallas.mosaic_gpu.planar_snake")(lin_idx, shape, minor_dim, ...) | Converts a linear index into an index into shape, trying to optimize locality. |

## Loop-like functions[\#](#loop-like-functions "Link to this heading")

|  |  |
|----|----|
| [`emit_pipeline`](_autosummary/jax.experimental.pallas.mosaic_gpu.emit_pipeline.html#jax.experimental.pallas.mosaic_gpu.emit_pipeline "jax.experimental.pallas.mosaic_gpu.emit_pipeline")(body, \*, grid\[, in_specs, ...\]) | Creates a function to emit a manual pipeline within a Pallas kernel. |
| [`emit_pipeline_warp_specialized`](_autosummary/jax.experimental.pallas.mosaic_gpu.emit_pipeline_warp_specialized.html#jax.experimental.pallas.mosaic_gpu.emit_pipeline_warp_specialized "jax.experimental.pallas.mosaic_gpu.emit_pipeline_warp_specialized")(body, \*, ...) | Creates a function to emit a warp-specialized pipeline. |
| [`nd_loop`](_autosummary/jax.experimental.pallas.mosaic_gpu.nd_loop.html#jax.experimental.pallas.mosaic_gpu.nd_loop "jax.experimental.pallas.mosaic_gpu.nd_loop")() | A loop over a multi-dimensional grid partitioned along the given axes. |
| [`dynamic_scheduling_loop`](_autosummary/jax.experimental.pallas.mosaic_gpu.dynamic_scheduling_loop.html#jax.experimental.pallas.mosaic_gpu.dynamic_scheduling_loop "jax.experimental.pallas.mosaic_gpu.dynamic_scheduling_loop")(\[init_carry\]) | A loop over program instances using dynamic work scheduling. |

## Synchronization[\#](#synchronization "Link to this heading")

|  |  |
|----|----|
| [`barrier_arrive`](_autosummary/jax.experimental.pallas.mosaic_gpu.barrier_arrive.html#jax.experimental.pallas.mosaic_gpu.barrier_arrive "jax.experimental.pallas.mosaic_gpu.barrier_arrive")(barrier) | Arrives at the given barrier. |
| [`barrier_wait`](_autosummary/jax.experimental.pallas.mosaic_gpu.barrier_wait.html#jax.experimental.pallas.mosaic_gpu.barrier_wait "jax.experimental.pallas.mosaic_gpu.barrier_wait")(barrier) | Waits on the given barrier. |
| [`semaphore_signal_parallel`](_autosummary/jax.experimental.pallas.mosaic_gpu.semaphore_signal_parallel.html#jax.experimental.pallas.mosaic_gpu.semaphore_signal_parallel "jax.experimental.pallas.mosaic_gpu.semaphore_signal_parallel")(\*signals) | Signals multiple semaphores without any guaranteed ordering of signal arrivals. |
| [`SemaphoreSignal`](_autosummary/jax.experimental.pallas.mosaic_gpu.SemaphoreSignal.html#jax.experimental.pallas.mosaic_gpu.SemaphoreSignal "jax.experimental.pallas.mosaic_gpu.SemaphoreSignal")(ref, \*, device_id\[, inc\]) |  |

## Asynchronous copies[\#](#asynchronous-copies "Link to this heading")

|  |  |
|----|----|
| [`commit_smem`](_autosummary/jax.experimental.pallas.mosaic_gpu.commit_smem.html#jax.experimental.pallas.mosaic_gpu.commit_smem "jax.experimental.pallas.mosaic_gpu.commit_smem")() | Commits all reads from/writes to SMEM, making them visible to TMA and MMA operations. |
| [`copy_gmem_to_smem`](_autosummary/jax.experimental.pallas.mosaic_gpu.copy_gmem_to_smem.html#jax.experimental.pallas.mosaic_gpu.copy_gmem_to_smem "jax.experimental.pallas.mosaic_gpu.copy_gmem_to_smem")(src, dst, barrier, \*\[, ...\]) | Asynchronously copies a GMEM reference to a SMEM reference. |
| [`copy_smem_to_gmem`](_autosummary/jax.experimental.pallas.mosaic_gpu.copy_smem_to_gmem.html#jax.experimental.pallas.mosaic_gpu.copy_smem_to_gmem "jax.experimental.pallas.mosaic_gpu.copy_smem_to_gmem")(src, dst\[, predicate, ...\]) | Asynchronously copies a SMEM reference to a GMEM reference. |
| [`wait_smem_to_gmem`](_autosummary/jax.experimental.pallas.mosaic_gpu.wait_smem_to_gmem.html#jax.experimental.pallas.mosaic_gpu.wait_smem_to_gmem "jax.experimental.pallas.mosaic_gpu.wait_smem_to_gmem")(n\[, wait_read_only\]) | Waits until no more than the most recent `n` SMEM-\>GMEM copies issued by the calling thread are in flight. |

## Hopper-specific functions[\#](#hopper-specific-functions "Link to this heading")

|  |  |
|----|----|
| [`wgmma`](_autosummary/jax.experimental.pallas.mosaic_gpu.wgmma.html#jax.experimental.pallas.mosaic_gpu.wgmma "jax.experimental.pallas.mosaic_gpu.wgmma")(acc, a, b) | Performs an asynchronous warp group matmul-accumulate on the given references. |
| [`wgmma_wait`](_autosummary/jax.experimental.pallas.mosaic_gpu.wgmma_wait.html#jax.experimental.pallas.mosaic_gpu.wgmma_wait "jax.experimental.pallas.mosaic_gpu.wgmma_wait")(n) | Waits until there is no more than `n` WGMMA operations in flight. |

## Blackwell-specific functions[\#](#blackwell-specific-functions "Link to this heading")

|  |  |
|----|----|
| [`tcgen05_mma`](_autosummary/jax.experimental.pallas.mosaic_gpu.tcgen05_mma.html#jax.experimental.pallas.mosaic_gpu.tcgen05_mma "jax.experimental.pallas.mosaic_gpu.tcgen05_mma")(acc, a, b\[, barrier, a_scale, ...\]) | Asynchronous matrix-multiply accumulate for TensorCore gen 5 (Blackwell). |
| [`tcgen05_commit_arrive`](_autosummary/jax.experimental.pallas.mosaic_gpu.tcgen05_commit_arrive.html#jax.experimental.pallas.mosaic_gpu.tcgen05_commit_arrive "jax.experimental.pallas.mosaic_gpu.tcgen05_commit_arrive")(barrier\[, collective_axis\]) | Tracks completion of all preceding `tcgen05_mma` and `async_copy_smem_to_tmem` calls. |
| [`async_load_tmem`](_autosummary/jax.experimental.pallas.mosaic_gpu.async_load_tmem.html#jax.experimental.pallas.mosaic_gpu.async_load_tmem "jax.experimental.pallas.mosaic_gpu.async_load_tmem")(src, \*\[, layout\]) | Performs an asynchronous load from the TMEM array. |
| [`async_store_tmem`](_autosummary/jax.experimental.pallas.mosaic_gpu.async_store_tmem.html#jax.experimental.pallas.mosaic_gpu.async_store_tmem "jax.experimental.pallas.mosaic_gpu.async_store_tmem")(ref, value) | Stores the value to TMEM. |
| [`wait_load_tmem`](_autosummary/jax.experimental.pallas.mosaic_gpu.wait_load_tmem.html#jax.experimental.pallas.mosaic_gpu.wait_load_tmem "jax.experimental.pallas.mosaic_gpu.wait_load_tmem")() | Awaits all previously asynchronous TMEM loads issued by the calling thread. |
| [`commit_tmem`](_autosummary/jax.experimental.pallas.mosaic_gpu.commit_tmem.html#jax.experimental.pallas.mosaic_gpu.commit_tmem "jax.experimental.pallas.mosaic_gpu.commit_tmem")() | Commits all writes to TMEM issued by the current thread. |
| [`try_cluster_cancel`](_autosummary/jax.experimental.pallas.mosaic_gpu.try_cluster_cancel.html#jax.experimental.pallas.mosaic_gpu.try_cluster_cancel "jax.experimental.pallas.mosaic_gpu.try_cluster_cancel")(result_ref, barrier) | Initiates an async request to claim a new work unit from the grid. |
| [`query_cluster_cancel`](_autosummary/jax.experimental.pallas.mosaic_gpu.query_cluster_cancel.html#jax.experimental.pallas.mosaic_gpu.query_cluster_cancel "jax.experimental.pallas.mosaic_gpu.query_cluster_cancel")(result_ref, grid_names) | Decodes the result of a `try_cluster_cancel` operation. |

## Multimem operations[\#](#multimem-operations "Link to this heading")

|  |  |
|----|----|
| [`multimem_store`](_autosummary/jax.experimental.pallas.mosaic_gpu.multimem_store.html#jax.experimental.pallas.mosaic_gpu.multimem_store "jax.experimental.pallas.mosaic_gpu.multimem_store")(source, ref, collective_axes) | Stores the value to ref on all devices present in collective_axes. |
| [`multimem_load_reduce`](_autosummary/jax.experimental.pallas.mosaic_gpu.multimem_load_reduce.html#jax.experimental.pallas.mosaic_gpu.multimem_load_reduce "jax.experimental.pallas.mosaic_gpu.multimem_load_reduce")(ref, \*, ...) | Loads from a GMEM reference on all devices present in collective_axes and reduces the loaded values. |

## Aliases[\#](#aliases "Link to this heading")

|  |  |
|----|----|
| [`ACC`](_autosummary/jax.experimental.pallas.mosaic_gpu.ACC.html#jax.experimental.pallas.mosaic_gpu.ACC "jax.experimental.pallas.mosaic_gpu.ACC") | alias of [`WGMMAAccumulatorRef`](_autosummary/jax.experimental.pallas.mosaic_gpu.WGMMAAccumulatorRef.html#jax.experimental.pallas.mosaic_gpu.WGMMAAccumulatorRef "jax._src.pallas.mosaic_gpu.core.WGMMAAccumulatorRef") |
| [`GMEM`](_autosummary/jax.experimental.pallas.mosaic_gpu.GMEM.html#jax.experimental.pallas.mosaic_gpu.GMEM "jax.experimental.pallas.mosaic_gpu.GMEM") | Alias of `jax.experimental.pallas.mosaic_gpu.MemorySpace.GMEM`. |
| [`SMEM`](_autosummary/jax.experimental.pallas.mosaic_gpu.SMEM.html#jax.experimental.pallas.mosaic_gpu.SMEM "jax.experimental.pallas.mosaic_gpu.SMEM") | Alias of `jax.experimental.pallas.mosaic_gpu.MemorySpace.SMEM`. |

[](_autosummary/jax.experimental.pallas.tpu.with_memory_space_constraint.html "previous page")

previous

jax.experimental.pallas.tpu.with_memory_space_constraint

[](_autosummary/jax.experimental.pallas.mosaic_gpu.Barrier.html "next page")

next

jax.experimental.pallas.mosaic_gpu.Barrier

Contents

- [Classes](#classes)
- [Functions](#functions)
- [Loop-like functions](#loop-like-functions)
- [Synchronization](#synchronization)
- [Asynchronous copies](#asynchronous-copies)
- [Hopper-specific functions](#hopper-specific-functions)
- [Blackwell-specific functions](#blackwell-specific-functions)
- [Multimem operations](#multimem-operations)
- [Aliases](#aliases)

By The JAX authors

© Copyright 2024, The JAX Authors.\
