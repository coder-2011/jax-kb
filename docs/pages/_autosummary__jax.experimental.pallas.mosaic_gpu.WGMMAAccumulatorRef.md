- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.mosaic_gpu` module](../jax.experimental.pallas.mosaic_gpu.html)
- jax.experimental.pallas.mosaic_gpu.WGMMAAccumulatorRef

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.mosaic_gpu.WGMMAAccumulatorRef.rst "Download source file")
-  .pdf

# jax.experimental.pallas.mosaic_gpu.WGMMAAccumulatorRef

## Contents

- [`WGMMAAccumulatorRef`](#jax.experimental.pallas.mosaic_gpu.WGMMAAccumulatorRef)
  - [`WGMMAAccumulatorRef.__init__()`](#jax.experimental.pallas.mosaic_gpu.WGMMAAccumulatorRef.__init__)

# jax.experimental.pallas.mosaic_gpu.WGMMAAccumulatorRef[\#](#jax-experimental-pallas-mosaic-gpu-wgmmaaccumulatorref "Link to this heading")

*class* jax.experimental.pallas.mosaic_gpu.WGMMAAccumulatorRef(*shape: 'tuple\[int*, *int\]'*, *dtype: 'jnp.dtype' = \<class 'jax.numpy.float32'\>*, *\_init: 'Any' = \<jax.\_src.state.types.Uninitialized object at 0x78348ba74fb0\>*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic_gpu/core.py#L1354-L1372)[\#](#jax.experimental.pallas.mosaic_gpu.WGMMAAccumulatorRef "Link to this definition")  
Parameters:  
- **shape** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*)

- **dtype** (*jnp.dtype*)

- **\_init** (*Any*)

\_\_init\_\_(*shape*, *dtype=\<class 'jax.numpy.float32'\>*, *\_init=\<jax.\_src.state.types.Uninitialized object\>*)[\#](#jax.experimental.pallas.mosaic_gpu.WGMMAAccumulatorRef.__init__ "Link to this definition")  
Parameters:  
- **shape** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*)

- **dtype** (*jnp.dtype*)

- **\_init** (*Any*)

Return type:  
None

Methods

|  |  |
|----|----|
| [`__init__`](#jax.experimental.pallas.mosaic_gpu.WGMMAAccumulatorRef.__init__ "jax.experimental.pallas.mosaic_gpu.WGMMAAccumulatorRef.__init__")(shape\[, dtype, \_init\]) |  |
| `get_ref_aval`() |  |
| `init`(array) |  |

Attributes

|         |     |
|---------|-----|
| `shape` |     |

[](jax.experimental.pallas.mosaic_gpu.TilingTransform.html "previous page")

previous

jax.experimental.pallas.mosaic_gpu.TilingTransform

[](jax.experimental.pallas.mosaic_gpu.as_torch_kernel.html "next page")

next

jax.experimental.pallas.mosaic_gpu.as_torch_kernel

Contents

- [`WGMMAAccumulatorRef`](#jax.experimental.pallas.mosaic_gpu.WGMMAAccumulatorRef)
  - [`WGMMAAccumulatorRef.__init__()`](#jax.experimental.pallas.mosaic_gpu.WGMMAAccumulatorRef.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
