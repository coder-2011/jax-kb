- [](index.html)
- [API Reference](jax.html)
- [`jax.experimental` module](jax.experimental.html)
- `jax.experimental.pallas` module

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](_sources/jax.experimental.pallas.rst "Download source file")
-  .pdf

# jax.experimental.pallas module

## Contents

- [Backends](#backends)
- [Classes](#classes)
- [Functions](#functions)
- [Synchronization](#synchronization)

# `jax.experimental.pallas` module[\#](#module-jax.experimental.pallas "Link to this heading")

Module for Pallas, a JAX extension for custom kernels.

See the Pallas documentation at [https://docs.jax.dev/en/latest/pallas/index.html](https://docs.jax.dev/en/latest/pallas/index.html).

## Backends[\#](#backends "Link to this heading")

- [Pallas TPU (TensorCore)](jax.experimental.pallas.tpu.html)
- [Pallas MGPU](jax.experimental.pallas.mosaic_gpu.html)
- [Triton](jax.experimental.pallas.triton.html)

## Classes[\#](#classes "Link to this heading")

|  |  |
|----|----|
| [`BlockSpec`](_autosummary/jax.experimental.pallas.BlockSpec.html#jax.experimental.pallas.BlockSpec "jax.experimental.pallas.BlockSpec")(\[block_shape, index_map, ...\]) | Specifies how an array should be sliced for each invocation of a kernel. |
| [`GridSpec`](_autosummary/jax.experimental.pallas.GridSpec.html#jax.experimental.pallas.GridSpec "jax.experimental.pallas.GridSpec")(\[grid, in_specs, out_specs, ...\]) | Encodes the grid parameters for [`jax.experimental.pallas.pallas_call()`](_autosummary/jax.experimental.pallas.pallas_call.html#jax.experimental.pallas.pallas_call "jax.experimental.pallas.pallas_call"). |
| [`Slice`](_autosummary/jax.experimental.pallas.Slice.html#jax.experimental.pallas.Slice "jax.experimental.pallas.Slice")(start, size\[, stride\]) | A slice with a start index and a size. |

## Functions[\#](#functions "Link to this heading")

|  |  |
|----|----|
| [`core_map`](_autosummary/jax.experimental.pallas.core_map.html#jax.experimental.pallas.core_map "jax.experimental.pallas.core_map")(mesh, \*\[, compiler_params, ...\]) | Runs a function on a mesh, mapping it over the devices in the mesh. |
| [`kernel`](_autosummary/jax.experimental.pallas.kernel.html#jax.experimental.pallas.kernel "jax.experimental.pallas.kernel")(\[body, out_type, scratch_types, ...\]) | Entry point for creating a Pallas kernel. |
| [`pallas_call`](_autosummary/jax.experimental.pallas.pallas_call.html#jax.experimental.pallas.pallas_call "jax.experimental.pallas.pallas_call")(kernel, out_shape, \*\[, ...\]) | Entry point for creating a Pallas kernel. |
| [`program_id`](_autosummary/jax.experimental.pallas.program_id.html#jax.experimental.pallas.program_id "jax.experimental.pallas.program_id")(axis) | Returns the kernel execution position along the given axis of the grid. |
| [`num_programs`](_autosummary/jax.experimental.pallas.num_programs.html#jax.experimental.pallas.num_programs "jax.experimental.pallas.num_programs")(axis) | Returns the size of the grid along the given axis. |
| [`cdiv`](_autosummary/jax.experimental.pallas.cdiv.html#jax.experimental.pallas.cdiv "jax.experimental.pallas.cdiv")() | Computes the ceiling division of a divided by b. |
| [`dslice`](_autosummary/jax.experimental.pallas.dslice.html#jax.experimental.pallas.dslice "jax.experimental.pallas.dslice")(start\[, size, stride\]) | Constructs a `Slice` from a start index and a size. |
| [`empty`](_autosummary/jax.experimental.pallas.empty.html#jax.experimental.pallas.empty "jax.experimental.pallas.empty")(shape, dtype, \*\[, out_sharding\]) | Create an empty array of possibly uninitialized values. |
| [`empty_like`](_autosummary/jax.experimental.pallas.empty_like.html#jax.experimental.pallas.empty_like "jax.experimental.pallas.empty_like")(x) | Create an empty PyTree of possibly uninitialized values. |
| [`broadcast_to`](_autosummary/jax.experimental.pallas.broadcast_to.html#jax.experimental.pallas.broadcast_to "jax.experimental.pallas.broadcast_to")(a, shape) | Broadcasts an array to a new shape. |
| [`debug_check`](_autosummary/jax.experimental.pallas.debug_check.html#jax.experimental.pallas.debug_check "jax.experimental.pallas.debug_check")(condition, message) | Check the condition if `enable_debug_checks()` is set, otherwise do nothing. |
| [`debug_print`](_autosummary/jax.experimental.pallas.debug_print.html#jax.experimental.pallas.debug_print "jax.experimental.pallas.debug_print")(fmt, \*args) | Prints values from inside a Pallas kernel. |
| [`dot`](_autosummary/jax.experimental.pallas.dot.html#jax.experimental.pallas.dot "jax.experimental.pallas.dot")(a, b\[, trans_a, trans_b, allow_tf32, ...\]) | Computes the dot product of two arrays. |
| [`get_global`](_autosummary/jax.experimental.pallas.get_global.html#jax.experimental.pallas.get_global "jax.experimental.pallas.get_global")(what) | Returns a global reference that persists across all kernel invocations. |
| [`loop`](_autosummary/jax.experimental.pallas.loop.html#jax.experimental.pallas.loop "jax.experimental.pallas.loop")() | Returns a decorator that calls the decorated function in a loop. |
| [`multiple_of`](_autosummary/jax.experimental.pallas.multiple_of.html#jax.experimental.pallas.multiple_of "jax.experimental.pallas.multiple_of")(x, values) | A compiler hint that asserts a value is a static multiple of another. |
| [`run_scoped`](_autosummary/jax.experimental.pallas.run_scoped.html#jax.experimental.pallas.run_scoped "jax.experimental.pallas.run_scoped")(f, \*types\[, collective_axes\]) | Calls the function with allocated references and returns the result. |
| [`when`](_autosummary/jax.experimental.pallas.when.html#jax.experimental.pallas.when "jax.experimental.pallas.when")(condition, /) | Calls the decorated function when the condition is met. |
| [`with_scoped`](_autosummary/jax.experimental.pallas.with_scoped.html#jax.experimental.pallas.with_scoped "jax.experimental.pallas.with_scoped")(\*types\[, collective_axes\]) | Returns a function decorator that runs a function with provided allocations. |

## Synchronization[\#](#synchronization "Link to this heading")

|  |  |
|----|----|
| [`semaphore_read`](_autosummary/jax.experimental.pallas.semaphore_read.html#jax.experimental.pallas.semaphore_read "jax.experimental.pallas.semaphore_read")(sem_or_view) | Reads the value of a semaphore. |
| [`semaphore_signal`](_autosummary/jax.experimental.pallas.semaphore_signal.html#jax.experimental.pallas.semaphore_signal "jax.experimental.pallas.semaphore_signal")(sem_or_view\[, inc, ...\]) | Increments the value of a semaphore. |
| [`semaphore_wait`](_autosummary/jax.experimental.pallas.semaphore_wait.html#jax.experimental.pallas.semaphore_wait "jax.experimental.pallas.semaphore_wait")(sem_or_view\[, value, decrement\]) | Blocks execution of the current thread until a semaphore reaches a value. |

[](_autosummary/jax.experimental.multihost_utils.global_array_to_host_local_array.html "previous page")

previous

jax.experimental.multihost_utils.global_array_to_host_local_array

[](jax.experimental.pallas.tpu.html "next page")

next

`jax.experimental.pallas.tpu` module

Contents

- [Backends](#backends)
- [Classes](#classes)
- [Functions](#functions)
- [Synchronization](#synchronization)

By The JAX authors

© Copyright 2024, The JAX Authors.\
