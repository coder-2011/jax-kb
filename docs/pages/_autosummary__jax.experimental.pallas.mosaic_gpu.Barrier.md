- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.mosaic_gpu` module](../jax.experimental.pallas.mosaic_gpu.html)
- jax.experimental.pallas.mosaic_gpu.Barrier

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.mosaic_gpu.Barrier.rst "Download source file")
-  .pdf

# jax.experimental.pallas.mosaic_gpu.Barrier

## Contents

- [`Barrier`](#jax.experimental.pallas.mosaic_gpu.Barrier)
  - [`Barrier.num_arrivals`](#jax.experimental.pallas.mosaic_gpu.Barrier.num_arrivals)
  - [`Barrier.num_barriers`](#jax.experimental.pallas.mosaic_gpu.Barrier.num_barriers)
  - [`Barrier.orders_tensor_core`](#jax.experimental.pallas.mosaic_gpu.Barrier.orders_tensor_core)
  - [`Barrier.__init__()`](#jax.experimental.pallas.mosaic_gpu.Barrier.__init__)

# jax.experimental.pallas.mosaic_gpu.Barrier[\#](#jax-experimental-pallas-mosaic-gpu-barrier "Link to this heading")

*class* jax.experimental.pallas.mosaic_gpu.Barrier(*\**, *num_arrivals=1*, *num_barriers=1*, *orders_tensor_core=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic_gpu/core.py#L1290-L1322)[\#](#jax.experimental.pallas.mosaic_gpu.Barrier "Link to this definition")  
Describes a barrier reference.

Parameters:  
- **num_arrivals** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **num_barriers** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*)

- **orders_tensor_core** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

num_arrivals[\#](#jax.experimental.pallas.mosaic_gpu.Barrier.num_arrivals "Link to this definition")  
The number of arrivals that will be recorded by this barrier.

Type:  
[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

num_barriers[\#](#jax.experimental.pallas.mosaic_gpu.Barrier.num_barriers "Link to this definition")  
The number of barriers that will be created. Individual barriers can be accessed by indexing into the barrier Ref.

Type:  
[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") \| Sequence\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")\]

orders_tensor_core[\#](#jax.experimental.pallas.mosaic_gpu.Barrier.orders_tensor_core "Link to this definition")  
If False, a successful wait from one thread does not guarantee that the TensorCore-related operations in other threads have completed. Similarly, when False any TensorCore operation in the waiting thread is allowed to begin before the wait succeeds.

Type:  
[bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")

\_\_init\_\_(*\**, *num_arrivals=1*, *num_barriers=1*, *orders_tensor_core=False*)[\#](#jax.experimental.pallas.mosaic_gpu.Barrier.__init__ "Link to this definition")  
Parameters:  
- **num_arrivals** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **num_barriers** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*)

- **orders_tensor_core** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

Return type:  
None

Methods

|  |  |
|----|----|
| [`__init__`](#jax.experimental.pallas.mosaic_gpu.Barrier.__init__ "jax.experimental.pallas.mosaic_gpu.Barrier.__init__")(\*\[, num_arrivals, num_barriers, ...\]) |  |
| `get_array_aval`() |  |
| `get_ref_aval`() |  |

Attributes

|  |  |
|----|----|
| [`num_arrivals`](#jax.experimental.pallas.mosaic_gpu.Barrier.num_arrivals "jax.experimental.pallas.mosaic_gpu.Barrier.num_arrivals") |  |
| [`num_barriers`](#jax.experimental.pallas.mosaic_gpu.Barrier.num_barriers "jax.experimental.pallas.mosaic_gpu.Barrier.num_barriers") |  |
| [`orders_tensor_core`](#jax.experimental.pallas.mosaic_gpu.Barrier.orders_tensor_core "jax.experimental.pallas.mosaic_gpu.Barrier.orders_tensor_core") |  |

[](../jax.experimental.pallas.mosaic_gpu.html "previous page")

previous

`jax.experimental.pallas.mosaic_gpu` module

[](jax.experimental.pallas.mosaic_gpu.BlockSpec.html "next page")

next

jax.experimental.pallas.mosaic_gpu.BlockSpec

Contents

- [`Barrier`](#jax.experimental.pallas.mosaic_gpu.Barrier)
  - [`Barrier.num_arrivals`](#jax.experimental.pallas.mosaic_gpu.Barrier.num_arrivals)
  - [`Barrier.num_barriers`](#jax.experimental.pallas.mosaic_gpu.Barrier.num_barriers)
  - [`Barrier.orders_tensor_core`](#jax.experimental.pallas.mosaic_gpu.Barrier.orders_tensor_core)
  - [`Barrier.__init__()`](#jax.experimental.pallas.mosaic_gpu.Barrier.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
