- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.mosaic_gpu` module](../jax.experimental.pallas.mosaic_gpu.html)
- jax.experimental.pallas.mosaic_gpu.Layout

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.mosaic_gpu.Layout.rst "Download source file")
-  .pdf

# jax.experimental.pallas.mosaic_gpu.Layout

## Contents

- [`Layout`](#jax.experimental.pallas.mosaic_gpu.Layout)
  - [`Layout.__init__()`](#jax.experimental.pallas.mosaic_gpu.Layout.__init__)

# jax.experimental.pallas.mosaic_gpu.Layout[\#](#jax-experimental-pallas-mosaic-gpu-layout "Link to this heading")

*class* jax.experimental.pallas.mosaic_gpu.Layout(*value*, *names=\<not given\>*, *\*values*, *module=None*, *qualname=None*, *type=None*, *start=1*, *boundary=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic_gpu/core.py#L1676-L1759)[\#](#jax.experimental.pallas.mosaic_gpu.Layout "Link to this definition")  
\_\_init\_\_(*\*args*, *\*\*kwds*)[\#](#jax.experimental.pallas.mosaic_gpu.Layout.__init__ "Link to this definition")  

Methods

|                               |     |
|-------------------------------|-----|
| `reduce`(axes)                |     |
| `to_mgpu`(\*args, \*\*kwargs) |     |

Attributes

|  |  |
|----|----|
| `WGMMA` | \[m, n\] matrix, where m % 64 == 0 == n % 8. |
| `WGMMA_8BIT` |  |
| `WGMMA_UPCAST_2X` |  |
| `WGMMA_UPCAST_4X` |  |
| `WGMMA_TRANSPOSED` |  |
| `WG_SPLAT` |  |
| `WG_STRIDED` |  |
| `TILED` |  |
| `TCGEN05` |  |
| `TCGEN05_TRANSPOSED` |  |
| `TCGEN05_M64_COLLECTIVE` |  |
| `TCGEN05_TMEM_NATIVE` |  |
| `TCGEN05_M64_COLLECTIVE_NATIVE` |  |
| `SMEM_GMEM_COPY` |  |
| `TMA_INDICES` |  |

[](jax.experimental.pallas.mosaic_gpu.MemorySpace.html "previous page")

previous

jax.experimental.pallas.mosaic_gpu.MemorySpace

[](jax.experimental.pallas.mosaic_gpu.SemaphoreType.html "next page")

next

jax.experimental.pallas.mosaic_gpu.SemaphoreType

Contents

- [`Layout`](#jax.experimental.pallas.mosaic_gpu.Layout)
  - [`Layout.__init__()`](#jax.experimental.pallas.mosaic_gpu.Layout.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
