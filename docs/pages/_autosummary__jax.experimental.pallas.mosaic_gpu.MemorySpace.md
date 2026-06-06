- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.mosaic_gpu` module](../jax.experimental.pallas.mosaic_gpu.html)
- jax.experimental.pallas.mosaic_gpu.MemorySpace

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.mosaic_gpu.MemorySpace.rst "Download source file")
-  .pdf

# jax.experimental.pallas.mosaic_gpu.MemorySpace

## Contents

- [`MemorySpace`](#jax.experimental.pallas.mosaic_gpu.MemorySpace)
  - [`MemorySpace.__init__()`](#jax.experimental.pallas.mosaic_gpu.MemorySpace.__init__)

# jax.experimental.pallas.mosaic_gpu.MemorySpace[\#](#jax-experimental-pallas-mosaic-gpu-memoryspace "Link to this heading")

*class* jax.experimental.pallas.mosaic_gpu.MemorySpace(*value*, *names=\<not given\>*, *\*values*, *module=None*, *qualname=None*, *type=None*, *start=1*, *boundary=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic_gpu/core.py#L146-L201)[\#](#jax.experimental.pallas.mosaic_gpu.MemorySpace "Link to this definition")  
\_\_init\_\_(*\*args*, *\*\*kwds*)[\#](#jax.experimental.pallas.mosaic_gpu.MemorySpace.__init__ "Link to this definition")  

Methods

|                          |     |
|--------------------------|-----|
| `like`(shape_dtype_like) |     |

Attributes

|        |                |
|--------|----------------|
| `GMEM` | Global memory. |
| `SMEM` | Shared memory. |
| `TMEM` | Tensor memory. |
| `REGS` | Registers.     |

[](jax.experimental.pallas.mosaic_gpu.CompilerParams.html "previous page")

previous

jax.experimental.pallas.mosaic_gpu.CompilerParams

[](jax.experimental.pallas.mosaic_gpu.Layout.html "next page")

next

jax.experimental.pallas.mosaic_gpu.Layout

Contents

- [`MemorySpace`](#jax.experimental.pallas.mosaic_gpu.MemorySpace)
  - [`MemorySpace.__init__()`](#jax.experimental.pallas.mosaic_gpu.MemorySpace.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
