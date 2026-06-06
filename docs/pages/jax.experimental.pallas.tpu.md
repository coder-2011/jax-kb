- [](index.html)
- [API Reference](jax.html)
- [`jax.experimental` module](jax.experimental.html)
- [`jax.experimental.pallas` module](jax.experimental.pallas.html)
- `jax.experimental.pallas.tpu` module

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](_sources/jax.experimental.pallas.tpu.rst "Download source file")
-  .pdf

# jax.experimental.pallas.tpu module

## Contents

- [Classes](#classes)
- [Functions](#functions)
- [Communication](#communication)
- [Pipelining](#pipelining)
- [Pseudorandom Number Generation](#pseudorandom-number-generation)
- [Interpret Mode](#interpret-mode)
- [Miscellaneous](#miscellaneous)

# `jax.experimental.pallas.tpu` module[\#](#module-jax.experimental.pallas.tpu "Link to this heading")

Mosaic-specific Pallas APIs.

## Classes[\#](#classes "Link to this heading")

|  |  |
|----|----|
| [`ChipVersion`](_autosummary/jax.experimental.pallas.tpu.ChipVersion.html#jax.experimental.pallas.tpu.ChipVersion "jax.experimental.pallas.tpu.ChipVersion")(value\[, names, module, ...\]) | TPU chip version. |
| [`CompilerParams`](_autosummary/jax.experimental.pallas.tpu.CompilerParams.html#jax.experimental.pallas.tpu.CompilerParams "jax.experimental.pallas.tpu.CompilerParams")(\[dimension_semantics, ...\]) | Mosaic TPU compiler parameters. |
| [`GridDimensionSemantics`](_autosummary/jax.experimental.pallas.tpu.GridDimensionSemantics.html#jax.experimental.pallas.tpu.GridDimensionSemantics "jax.experimental.pallas.tpu.GridDimensionSemantics")(value\[, names, ...\]) |  |
| [`MemorySpace`](_autosummary/jax.experimental.pallas.tpu.MemorySpace.html#jax.experimental.pallas.tpu.MemorySpace "jax.experimental.pallas.tpu.MemorySpace")(value\[, names, module, ...\]) |  |
| [`PrefetchScalarGridSpec`](_autosummary/jax.experimental.pallas.tpu.PrefetchScalarGridSpec.html#jax.experimental.pallas.tpu.PrefetchScalarGridSpec "jax.experimental.pallas.tpu.PrefetchScalarGridSpec")(num_scalar_prefetch) |  |
| [`SemaphoreType`](_autosummary/jax.experimental.pallas.tpu.SemaphoreType.html#jax.experimental.pallas.tpu.SemaphoreType "jax.experimental.pallas.tpu.SemaphoreType")(value\[, names, module, ...\]) |  |
| [`TpuInfo`](_autosummary/jax.experimental.pallas.tpu.TpuInfo.html#jax.experimental.pallas.tpu.TpuInfo "jax.experimental.pallas.tpu.TpuInfo")(\*, chip_version, generation, ...\[, ...\]) | TPU hardware information. |

## Functions[\#](#functions "Link to this heading")

|  |  |
|----|----|
| [`load`](_autosummary/jax.experimental.pallas.tpu.load.html#jax.experimental.pallas.tpu.load "jax.experimental.pallas.tpu.load")(ref, \*\[, mask\]) | Loads an array from the given ref. |
| [`store`](_autosummary/jax.experimental.pallas.tpu.store.html#jax.experimental.pallas.tpu.store "jax.experimental.pallas.tpu.store")(ref, val, \*\[, mask\]) | Stores a value to the given ref. |

## Communication[\#](#communication "Link to this heading")

|  |  |
|----|----|
| [`async_copy`](_autosummary/jax.experimental.pallas.tpu.async_copy.html#jax.experimental.pallas.tpu.async_copy "jax.experimental.pallas.tpu.async_copy")(src_ref, dst_ref, sem, \*\[, ...\]) | Issues a DMA copying from src_ref to dst_ref. |
| [`async_remote_copy`](_autosummary/jax.experimental.pallas.tpu.async_remote_copy.html#jax.experimental.pallas.tpu.async_remote_copy "jax.experimental.pallas.tpu.async_remote_copy")(src_ref, dst_ref, ...\[, ...\]) | Issues a remote DMA copying from src_ref to dst_ref. |
| [`make_async_copy`](_autosummary/jax.experimental.pallas.tpu.make_async_copy.html#jax.experimental.pallas.tpu.make_async_copy "jax.experimental.pallas.tpu.make_async_copy")(src_ref, dst_ref, sem) | Creates a description of an asynchronous copy operation. |
| [`make_async_remote_copy`](_autosummary/jax.experimental.pallas.tpu.make_async_remote_copy.html#jax.experimental.pallas.tpu.make_async_remote_copy "jax.experimental.pallas.tpu.make_async_remote_copy")(src_ref, dst_ref, ...) | Creates a description of a remote copy operation. |
| [`sync_copy`](_autosummary/jax.experimental.pallas.tpu.sync_copy.html#jax.experimental.pallas.tpu.sync_copy "jax.experimental.pallas.tpu.sync_copy")(src_ref, dst_ref, \*\[, add\]) | Synchronously copies a PyTree of refs to another PyTree of refs. |

## Pipelining[\#](#pipelining "Link to this heading")

|  |  |
|----|----|
| [`BufferedRef`](_autosummary/jax.experimental.pallas.tpu.BufferedRef.html#jax.experimental.pallas.tpu.BufferedRef "jax.experimental.pallas.tpu.BufferedRef")(\_spec, \_buffer_type, ...\[, ...\]) | A helper class to automate VMEM double buffering in pallas pipelines. |
| [`BufferedRefBase`](_autosummary/jax.experimental.pallas.tpu.BufferedRefBase.html#jax.experimental.pallas.tpu.BufferedRefBase "jax.experimental.pallas.tpu.BufferedRefBase")() | Abstract interface for BufferedRefs. |
| [`emit_pipeline`](_autosummary/jax.experimental.pallas.tpu.emit_pipeline.html#jax.experimental.pallas.tpu.emit_pipeline "jax.experimental.pallas.tpu.emit_pipeline")(body, \*, grid\[, in_specs, ...\]) | Creates a function to emit a manual pallas pipeline. |
| [`emit_pipeline_with_allocations`](_autosummary/jax.experimental.pallas.tpu.emit_pipeline_with_allocations.html#jax.experimental.pallas.tpu.emit_pipeline_with_allocations "jax.experimental.pallas.tpu.emit_pipeline_with_allocations")(body, \*, grid) | Creates pallas pipeline and top-level allocation preparation functions. |

## Pseudorandom Number Generation[\#](#pseudorandom-number-generation "Link to this heading")

|  |  |
|----|----|
| [`prng_seed`](_autosummary/jax.experimental.pallas.tpu.prng_seed.html#jax.experimental.pallas.tpu.prng_seed "jax.experimental.pallas.tpu.prng_seed")(\*seeds) | Sets the seed for PRNG. |
| [`sample_block`](_autosummary/jax.experimental.pallas.tpu.sample_block.html#jax.experimental.pallas.tpu.sample_block "jax.experimental.pallas.tpu.sample_block")(sampler_fn, global_key, ...\[, ...\]) | Samples a block of random values with invariance guarantees. |
| [`stateful_bernoulli`](_autosummary/jax.experimental.pallas.tpu.stateful_bernoulli.html#jax.experimental.pallas.tpu.stateful_bernoulli "jax.experimental.pallas.tpu.stateful_bernoulli")(\*args, \*\*kwargs) | Sample Bernoulli random values with given shape and mean. |
| [`stateful_bits`](_autosummary/jax.experimental.pallas.tpu.stateful_bits.html#jax.experimental.pallas.tpu.stateful_bits "jax.experimental.pallas.tpu.stateful_bits")(\*args, \*\*kwargs) | Sample uniform bits in the form of unsigned integers. |
| [`stateful_normal`](_autosummary/jax.experimental.pallas.tpu.stateful_normal.html#jax.experimental.pallas.tpu.stateful_normal "jax.experimental.pallas.tpu.stateful_normal")(\*args, \*\*kwargs) | Sample standard normal random values with given shape and float dtype. |
| [`stateful_uniform`](_autosummary/jax.experimental.pallas.tpu.stateful_uniform.html#jax.experimental.pallas.tpu.stateful_uniform "jax.experimental.pallas.tpu.stateful_uniform")(\*args, \*\*kwargs) | Sample uniform random values in \[minval, maxval) with given shape/dtype. |
| [`to_pallas_key`](_autosummary/jax.experimental.pallas.tpu.to_pallas_key.html#jax.experimental.pallas.tpu.to_pallas_key "jax.experimental.pallas.tpu.to_pallas_key")(key) | Helper function for converting non-Pallas PRNG keys into Pallas keys. |

## Interpret Mode[\#](#interpret-mode "Link to this heading")

|  |  |
|----|----|
| [`force_tpu_interpret_mode`](_autosummary/jax.experimental.pallas.tpu.force_tpu_interpret_mode.html#jax.experimental.pallas.tpu.force_tpu_interpret_mode "jax.experimental.pallas.tpu.force_tpu_interpret_mode")(\[params\]) | Context manager that forces TPU interpret mode under its dynamic context. |
| [`InterpretParams`](_autosummary/jax.experimental.pallas.tpu.InterpretParams.html#jax.experimental.pallas.tpu.InterpretParams "jax.experimental.pallas.tpu.InterpretParams")(\*\[, detect_races, ...\]) | Parameters for TPU interpret mode. |
| [`reset_tpu_interpret_mode_state`](_autosummary/jax.experimental.pallas.tpu.reset_tpu_interpret_mode_state.html#jax.experimental.pallas.tpu.reset_tpu_interpret_mode_state "jax.experimental.pallas.tpu.reset_tpu_interpret_mode_state")() | Resets all global, shared state used by TPU interpret mode. |
| [`set_tpu_interpret_mode`](_autosummary/jax.experimental.pallas.tpu.set_tpu_interpret_mode.html#jax.experimental.pallas.tpu.set_tpu_interpret_mode "jax.experimental.pallas.tpu.set_tpu_interpret_mode")(\[params\]) |  |

## Miscellaneous[\#](#miscellaneous "Link to this heading")

|  |  |
|----|----|
| [`core_barrier`](_autosummary/jax.experimental.pallas.tpu.core_barrier.html#jax.experimental.pallas.tpu.core_barrier "jax.experimental.pallas.tpu.core_barrier")(sem, \*, core_axis_name) | Synchronizes all cores in a given axis. |
| [`get_barrier_semaphore`](_autosummary/jax.experimental.pallas.tpu.get_barrier_semaphore.html#jax.experimental.pallas.tpu.get_barrier_semaphore "jax.experimental.pallas.tpu.get_barrier_semaphore")() | Returns a barrier semaphore. |
| [`get_tpu_info`](_autosummary/jax.experimental.pallas.tpu.get_tpu_info.html#jax.experimental.pallas.tpu.get_tpu_info "jax.experimental.pallas.tpu.get_tpu_info")() | Returns the TPU hardware info for the current device. |
| [`is_tpu_device`](_autosummary/jax.experimental.pallas.tpu.is_tpu_device.html#jax.experimental.pallas.tpu.is_tpu_device "jax.experimental.pallas.tpu.is_tpu_device")() |  |
| [`run_on_first_core`](_autosummary/jax.experimental.pallas.tpu.run_on_first_core.html#jax.experimental.pallas.tpu.run_on_first_core "jax.experimental.pallas.tpu.run_on_first_core")(core_axis_name) | Runs a function on the first core in a given axis. |
| [`with_memory_space_constraint`](_autosummary/jax.experimental.pallas.tpu.with_memory_space_constraint.html#jax.experimental.pallas.tpu.with_memory_space_constraint "jax.experimental.pallas.tpu.with_memory_space_constraint")(x, memory_space) | Constrains the memory space of an array. |

[](jax.experimental.pallas.html "previous page")

previous

`jax.experimental.pallas` module

[](_autosummary/jax.experimental.pallas.tpu.ChipVersion.html "next page")

next

jax.experimental.pallas.tpu.ChipVersion

Contents

- [Classes](#classes)
- [Functions](#functions)
- [Communication](#communication)
- [Pipelining](#pipelining)
- [Pseudorandom Number Generation](#pseudorandom-number-generation)
- [Interpret Mode](#interpret-mode)
- [Miscellaneous](#miscellaneous)

By The JAX authors

© Copyright 2024, The JAX Authors.\
