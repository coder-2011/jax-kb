- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.mosaic_gpu` module](../jax.experimental.pallas.mosaic_gpu.html)
- jax.experimental.pallas.mosaic_gpu.BlockSpec

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.mosaic_gpu.BlockSpec.rst "Download source file")
-  .pdf

# jax.experimental.pallas.mosaic_gpu.BlockSpec

## Contents

- [`BlockSpec`](#jax.experimental.pallas.mosaic_gpu.BlockSpec)
  - [`BlockSpec.transforms`](#jax.experimental.pallas.mosaic_gpu.BlockSpec.transforms)
  - [`BlockSpec.delay_release`](#jax.experimental.pallas.mosaic_gpu.BlockSpec.delay_release)
  - [`BlockSpec.collective_axes`](#jax.experimental.pallas.mosaic_gpu.BlockSpec.collective_axes)
  - [`BlockSpec.__init__()`](#jax.experimental.pallas.mosaic_gpu.BlockSpec.__init__)

# jax.experimental.pallas.mosaic_gpu.BlockSpec[\#](#jax-experimental-pallas-mosaic-gpu-blockspec "Link to this heading")

*class* jax.experimental.pallas.mosaic_gpu.BlockSpec(*block_shape=None*, *index_map=None*, *pipeline_mode=None*, *transforms=()*, *delay_release=0*, *collective_axes=()*, *\**, *memory_space=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic_gpu/core.py#L1198-L1252)[\#](#jax.experimental.pallas.mosaic_gpu.BlockSpec "Link to this definition")  
A GPU-specific `BlockSpec`.

Parameters:  
- **block_shape** (*Sequence\[BlockDim* *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None\]* *\|* *None*)

- **index_map** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[...,* *Any\]* *\|* *None*)

- **pipeline_mode** (*Buffered* *\|* *None*)

- **transforms** (*Sequence\[state_types.Transform\]*)

- **delay_release** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **collective_axes** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[Hashable,* *...\]*)

- **memory_space** (*Any* *\|* *None*)

transforms[\#](#jax.experimental.pallas.mosaic_gpu.BlockSpec.transforms "Link to this definition")  
A sequence of transforms that will be applied to the reference.

Type:  
Sequence\[state_types.Transform\]

delay_release[\#](#jax.experimental.pallas.mosaic_gpu.BlockSpec.delay_release "Link to this definition")  
used during pipelining to delay the release of resources of a slot after it is used in the computation.

Type:  
[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

collective_axes[\#](#jax.experimental.pallas.mosaic_gpu.BlockSpec.collective_axes "Link to this definition")  
When set, all blocks along the specified axes must execute the same sequence of pipeline operations (with the only exception being the index_map in non-collective `BlockSpec`s), and all of them must return the same block from the index_map for this operand. This enables the pipelining helpers to use collective async copies, which can improve performance.

Type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[Hashable, …\]

\_\_init\_\_(*block_shape=None*, *index_map=None*, *pipeline_mode=None*, *transforms=()*, *delay_release=0*, *collective_axes=()*, *\**, *memory_space=None*)[\#](#jax.experimental.pallas.mosaic_gpu.BlockSpec.__init__ "Link to this definition")  
Parameters:  
- **block_shape** (*Sequence\[BlockDim* *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None\]* *\|* *None*)

- **index_map** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[...,* *Any\]* *\|* *None*)

- **pipeline_mode** (*Buffered* *\|* *None*)

- **transforms** (*Sequence\[state_types.Transform\]*)

- **delay_release** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **collective_axes** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[Hashable,* *...\]*)

- **memory_space** (*Any* *\|* *None*)

Return type:  
None

Methods

|  |  |
|----|----|
| [`__init__`](#jax.experimental.pallas.mosaic_gpu.BlockSpec.__init__ "jax.experimental.pallas.mosaic_gpu.BlockSpec.__init__")(\[block_shape, index_map, ...\]) |  |
| `replace`(\*\*changes) | Return a new object replacing specified fields with new values. |
| `to_block_mapping`(origin, array_aval, \*, ...) |  |

Attributes

|  |  |
|----|----|
| `block_shape` |  |
| [`collective_axes`](#jax.experimental.pallas.mosaic_gpu.BlockSpec.collective_axes "jax.experimental.pallas.mosaic_gpu.BlockSpec.collective_axes") |  |
| [`delay_release`](#jax.experimental.pallas.mosaic_gpu.BlockSpec.delay_release "jax.experimental.pallas.mosaic_gpu.BlockSpec.delay_release") |  |
| `index_map` |  |
| `memory_space` |  |
| `pipeline_mode` |  |
| [`transforms`](#jax.experimental.pallas.mosaic_gpu.BlockSpec.transforms "jax.experimental.pallas.mosaic_gpu.BlockSpec.transforms") |  |

[](jax.experimental.pallas.mosaic_gpu.Barrier.html "previous page")

previous

jax.experimental.pallas.mosaic_gpu.Barrier

[](jax.experimental.pallas.mosaic_gpu.CompilerParams.html "next page")

next

jax.experimental.pallas.mosaic_gpu.CompilerParams

Contents

- [`BlockSpec`](#jax.experimental.pallas.mosaic_gpu.BlockSpec)
  - [`BlockSpec.transforms`](#jax.experimental.pallas.mosaic_gpu.BlockSpec.transforms)
  - [`BlockSpec.delay_release`](#jax.experimental.pallas.mosaic_gpu.BlockSpec.delay_release)
  - [`BlockSpec.collective_axes`](#jax.experimental.pallas.mosaic_gpu.BlockSpec.collective_axes)
  - [`BlockSpec.__init__()`](#jax.experimental.pallas.mosaic_gpu.BlockSpec.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
