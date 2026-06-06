- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.tpu` module](../jax.experimental.pallas.tpu.html)
- jax.experimental.pallas.tpu.InterpretParams

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.tpu.InterpretParams.rst "Download source file")
-  .pdf

# jax.experimental.pallas.tpu.InterpretParams

## Contents

- [`InterpretParams`](#jax.experimental.pallas.tpu.InterpretParams)
  - [`InterpretParams.dma_execution_mode`](#jax.experimental.pallas.tpu.InterpretParams.dma_execution_mode)
  - [`InterpretParams.random_seed`](#jax.experimental.pallas.tpu.InterpretParams.random_seed)
  - [`InterpretParams.grid_point_recorder`](#jax.experimental.pallas.tpu.InterpretParams.grid_point_recorder)
  - [`InterpretParams.allow_hbm_allocation_in_run_scoped`](#jax.experimental.pallas.tpu.InterpretParams.allow_hbm_allocation_in_run_scoped)
  - [`InterpretParams.buffer_bounds`](#jax.experimental.pallas.tpu.InterpretParams.buffer_bounds)
  - [`InterpretParams.__init__()`](#jax.experimental.pallas.tpu.InterpretParams.__init__)

# jax.experimental.pallas.tpu.InterpretParams[\#](#jax-experimental-pallas-tpu-interpretparams "Link to this heading")

*class* jax.experimental.pallas.tpu.InterpretParams(*\**, *detect_races=False*, *out_of_bounds_reads='raise'*, *skip_floating_point_ops=False*, *uninitialized_memory='nan'*, *num_cores_or_threads=1*, *vector_clock_size=None*, *logging_mode=None*, *dma_execution_mode='on_wait'*, *random_seed=None*, *grid_point_recorder=None*, *allow_hbm_allocation_in_run_scoped=False*, *buffer_bounds='logical'*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic/interpret/params.py#L120-L189)[\#](#jax.experimental.pallas.tpu.InterpretParams "Link to this definition")  
Parameters for TPU interpret mode.

TPU interpret mode is a way run Pallas TPU kernels on CPU, while simulating a TPU’s shared memory (HBM, VMEM, etc.), communication (remote and local DMAs), and synchronization operations (semaphores, barriers, etc.). This mode is intended for debugging and testing.

To run a kernel under TPU interpret mode, pass an instance of `InterpretParams` as an argument for the `interpret` parameter of [`jax.experimental.pallas.pallas_call()`](jax.experimental.pallas.pallas_call.html#jax.experimental.pallas.pallas_call "jax.experimental.pallas.pallas_call") or [`jax.experimental.pallas.core_map()`](jax.experimental.pallas.core_map.html#jax.experimental.pallas.core_map "jax.experimental.pallas.core_map").

NOTE: If an exception is raised while interpreting a kernel, you must call [`reset_tpu_interpret_mode_state()`](jax.experimental.pallas.tpu.reset_tpu_interpret_mode_state.html#jax.experimental.pallas.tpu.reset_tpu_interpret_mode_state "jax.experimental.pallas.tpu.reset_tpu_interpret_mode_state") before using TPU interpret mode again in the same process.

Parameters:  
- **detect_races** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

- **out_of_bounds_reads** ([*Literal*](https://docs.python.org/3/library/typing.html#typing.Literal "(in Python v3.14)")*\['raise',* *'uninitialized'\]*)

- **skip_floating_point_ops** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

- **uninitialized_memory** ([*Literal*](https://docs.python.org/3/library/typing.html#typing.Literal "(in Python v3.14)")*\['nan',* *'zero'\]*)

- **num_cores_or_threads** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **vector_clock_size** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*)

- **logging_mode** (*LoggingMode* *\|* *None*)

- **dma_execution_mode** ([*Literal*](https://docs.python.org/3/library/typing.html#typing.Literal "(in Python v3.14)")*\['eager',* *'on_wait'\]*)

- **random_seed** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*)

- **grid_point_recorder** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "collections.abc.Callable")*\[\[*[*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.14)")*,* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*int32*](jax.numpy.int32.html#jax.numpy.int32 "jax.numpy.int32")*,* *...\],* [*int32*](jax.numpy.int32.html#jax.numpy.int32 "jax.numpy.int32")*\],* [*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.14)")*\]* *\|* *None*)

- **allow_hbm_allocation_in_run_scoped** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

- **buffer_bounds** ([*Literal*](https://docs.python.org/3/library/typing.html#typing.Literal "(in Python v3.14)")*\['logical',* *'padded'\]*)

dma_execution_mode[\#](#jax.experimental.pallas.tpu.InterpretParams.dma_execution_mode "Link to this definition")  
If “eager”, DMAs are executed as soon as they are issued. If “on_wait”, DMA reads or writes are only executed when a device is waiting on a DMA semaphore that will be signaled when the read or write is complete. Default: “on_wait”.

Type:  
[Literal](jax.extend.core.Literal.html#jax.extend.core.Literal "jax.extend.core.Literal")\[‘eager’, ‘on_wait’\]

random_seed[\#](#jax.experimental.pallas.tpu.InterpretParams.random_seed "Link to this definition")  
Seed for random number generator used during interpretation. Currently random numbers are used to randomize the grid coordinates along dimensions with ‘parallel’ semantics. Default: None.

Type:  
[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") \| None

grid_point_recorder[\#](#jax.experimental.pallas.tpu.InterpretParams.grid_point_recorder "Link to this definition")  
Callback that is invoked by the interpreter for each grid point in the order in which the grid points are traversed. The callback is invoked with two arguments: - A tuple of grid coordinates. - The local core ID of the core that is processing the grid point. This callback is intended for inspecting - the randomization of coordinates along grid dimensions with ‘parallel’ semantics and - the mapping of grid points to local (i.e. per-device) cores. Default: None.

Type:  
[collections.abc.Callable](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "collections.abc.Callable")\[\[Any, [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[numpy.int32](jax.numpy.int32.html#jax.numpy.int32 "jax.numpy.int32"), …\], [numpy.int32](jax.numpy.int32.html#jax.numpy.int32 "jax.numpy.int32")\], Any\] \| None

allow_hbm_allocation_in_run_scoped[\#](#jax.experimental.pallas.tpu.InterpretParams.allow_hbm_allocation_in_run_scoped "Link to this definition")  
If True, allows the allocation of HBM buffers (which are then shared across the cores in a device) in run_scoped. While this behavior can be enabled in the interpreter, allocating HBM buffers with run_scoped is not supported when executing Pallas kernels on a real TPU. Default: False.

Type:  
[bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")

buffer_bounds[\#](#jax.experimental.pallas.tpu.InterpretParams.buffer_bounds "Link to this definition")  
If “padded”, reads and writes of a buffer are only considered out-of-bounds when they go beyond the ‘padded shape’ of a buffer. The amount of padding is determined by the TPU device kind that is being simulated by TPU interpret mode (to be set with jax.sharding.use_abstract_mesh in the context from where the interpreter Pallas kernel is called). Any part of a read that is outside of the buffer’s shape but inside the padded shape returns uninitialized values (see the “uninitialized_memory” attribute of the superclass SharedInterpretParams). Any part of a write that is outside of the buffer’s shape but inside the padded shape is ignored. If “logical”, reads and writes are considered out-of-bounds when outside of the buffer’s logical shape. Default: “logical”.

Type:  
[Literal](jax.extend.core.Literal.html#jax.extend.core.Literal "jax.extend.core.Literal")\[‘logical’, ‘padded’\]

\_\_init\_\_(*\**, *detect_races=False*, *out_of_bounds_reads='raise'*, *skip_floating_point_ops=False*, *uninitialized_memory='nan'*, *num_cores_or_threads=1*, *vector_clock_size=None*, *logging_mode=None*, *dma_execution_mode='on_wait'*, *random_seed=None*, *grid_point_recorder=None*, *allow_hbm_allocation_in_run_scoped=False*, *buffer_bounds='logical'*)[\#](#jax.experimental.pallas.tpu.InterpretParams.__init__ "Link to this definition")  
Parameters:  
- **detect_races** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

- **out_of_bounds_reads** ([*Literal*](https://docs.python.org/3/library/typing.html#typing.Literal "(in Python v3.14)")*\['raise',* *'uninitialized'\]*)

- **skip_floating_point_ops** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

- **uninitialized_memory** ([*Literal*](https://docs.python.org/3/library/typing.html#typing.Literal "(in Python v3.14)")*\['nan',* *'zero'\]*)

- **num_cores_or_threads** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **vector_clock_size** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*)

- **logging_mode** (*LoggingMode* *\|* *None*)

- **dma_execution_mode** ([*Literal*](https://docs.python.org/3/library/typing.html#typing.Literal "(in Python v3.14)")*\['eager',* *'on_wait'\]*)

- **random_seed** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*)

- **grid_point_recorder** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "collections.abc.Callable")*\[\[*[*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.14)")*,* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*int32*](jax.numpy.int32.html#jax.numpy.int32 "jax.numpy.int32")*,* *...\],* [*int32*](jax.numpy.int32.html#jax.numpy.int32 "jax.numpy.int32")*\],* [*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.14)")*\]* *\|* *None*)

- **allow_hbm_allocation_in_run_scoped** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

- **buffer_bounds** ([*Literal*](https://docs.python.org/3/library/typing.html#typing.Literal "(in Python v3.14)")*\['logical',* *'padded'\]*)

Return type:  
None

Methods

|  |  |
|----|----|
| [`__init__`](#jax.experimental.pallas.tpu.InterpretParams.__init__ "jax.experimental.pallas.tpu.InterpretParams.__init__")(\*\[, detect_races, ...\]) |  |
| `get_vector_clock_size`(num_devices) | Returns the number of vector clocks to use for TPU interpret mode.\` |

Attributes

|  |  |
|----|----|
| [`allow_hbm_allocation_in_run_scoped`](#jax.experimental.pallas.tpu.InterpretParams.allow_hbm_allocation_in_run_scoped "jax.experimental.pallas.tpu.InterpretParams.allow_hbm_allocation_in_run_scoped") |  |
| [`buffer_bounds`](#jax.experimental.pallas.tpu.InterpretParams.buffer_bounds "jax.experimental.pallas.tpu.InterpretParams.buffer_bounds") |  |
| `detect_races` |  |
| [`dma_execution_mode`](#jax.experimental.pallas.tpu.InterpretParams.dma_execution_mode "jax.experimental.pallas.tpu.InterpretParams.dma_execution_mode") |  |
| [`grid_point_recorder`](#jax.experimental.pallas.tpu.InterpretParams.grid_point_recorder "jax.experimental.pallas.tpu.InterpretParams.grid_point_recorder") |  |
| `logging_mode` |  |
| `num_cores_or_threads` |  |
| `num_cores_per_device` |  |
| `out_of_bounds_reads` |  |
| [`random_seed`](#jax.experimental.pallas.tpu.InterpretParams.random_seed "jax.experimental.pallas.tpu.InterpretParams.random_seed") |  |
| `skip_floating_point_ops` |  |
| `uninitialized_memory` |  |
| `vector_clock_size` |  |

[](jax.experimental.pallas.tpu.force_tpu_interpret_mode.html "previous page")

previous

jax.experimental.pallas.tpu.force_tpu_interpret_mode

[](jax.experimental.pallas.tpu.reset_tpu_interpret_mode_state.html "next page")

next

jax.experimental.pallas.tpu.reset_tpu_interpret_mode_state

Contents

- [`InterpretParams`](#jax.experimental.pallas.tpu.InterpretParams)
  - [`InterpretParams.dma_execution_mode`](#jax.experimental.pallas.tpu.InterpretParams.dma_execution_mode)
  - [`InterpretParams.random_seed`](#jax.experimental.pallas.tpu.InterpretParams.random_seed)
  - [`InterpretParams.grid_point_recorder`](#jax.experimental.pallas.tpu.InterpretParams.grid_point_recorder)
  - [`InterpretParams.allow_hbm_allocation_in_run_scoped`](#jax.experimental.pallas.tpu.InterpretParams.allow_hbm_allocation_in_run_scoped)
  - [`InterpretParams.buffer_bounds`](#jax.experimental.pallas.tpu.InterpretParams.buffer_bounds)
  - [`InterpretParams.__init__()`](#jax.experimental.pallas.tpu.InterpretParams.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
