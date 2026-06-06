- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.mosaic_gpu` module](../jax.experimental.pallas.mosaic_gpu.html)
- jax.experimental.pallas.mosaic_gpu.SemaphoreSignal

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.mosaic_gpu.SemaphoreSignal.rst "Download source file")
-  .pdf

# jax.experimental.pallas.mosaic_gpu.SemaphoreSignal

## Contents

- [`SemaphoreSignal`](#jax.experimental.pallas.mosaic_gpu.SemaphoreSignal)
  - [`SemaphoreSignal.__init__()`](#jax.experimental.pallas.mosaic_gpu.SemaphoreSignal.__init__)

# jax.experimental.pallas.mosaic_gpu.SemaphoreSignal[\#](#jax-experimental-pallas-mosaic-gpu-semaphoresignal "Link to this heading")

*class* jax.experimental.pallas.mosaic_gpu.SemaphoreSignal(*ref: '\_Ref'*, *\**, *device_id: 'pallas_primitives.DeviceId \| None'*, *inc: 'int \| jax.Array' = 1*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic_gpu/primitives.py#L4168-L4174)[\#](#jax.experimental.pallas.mosaic_gpu.SemaphoreSignal "Link to this definition")  
Parameters:  
- **ref** (*\_Ref*)

- **device_id** (*pallas_primitives.DeviceId* *\|* *None*)

- **inc** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*jax.Array*](jax.Array.html#jax.Array "jax.Array"))

\_\_init\_\_(*ref*, *\**, *device_id*, *inc=1*)[\#](#jax.experimental.pallas.mosaic_gpu.SemaphoreSignal.__init__ "Link to this definition")  
Parameters:  
- **ref** (*\_Ref*)

- **device_id** (*pallas_primitives.DeviceId* *\|* *None*)

- **inc** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*jax.Array*](jax.Array.html#jax.Array "jax.Array"))

Return type:  
None

Methods

|  |  |
|----|----|
| [`__init__`](#jax.experimental.pallas.mosaic_gpu.SemaphoreSignal.__init__ "jax.experimental.pallas.mosaic_gpu.SemaphoreSignal.__init__")(ref, \*, device_id\[, inc\]) |  |

Attributes

|             |     |
|-------------|-----|
| `inc`       |     |
| `ref`       |     |
| `device_id` |     |

[](jax.experimental.pallas.mosaic_gpu.semaphore_signal_parallel.html "previous page")

previous

jax.experimental.pallas.mosaic_gpu.semaphore_signal_parallel

[](jax.experimental.pallas.mosaic_gpu.commit_smem.html "next page")

next

jax.experimental.pallas.mosaic_gpu.commit_smem

Contents

- [`SemaphoreSignal`](#jax.experimental.pallas.mosaic_gpu.SemaphoreSignal)
  - [`SemaphoreSignal.__init__()`](#jax.experimental.pallas.mosaic_gpu.SemaphoreSignal.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
