- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.mosaic_gpu` module](../jax.experimental.pallas.mosaic_gpu.html)
- jax.experimental.pallas.mosaic_gpu.TilingTransform

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.mosaic_gpu.TilingTransform.rst "Download source file")
-  .pdf

# jax.experimental.pallas.mosaic_gpu.TilingTransform

## Contents

- [`TilingTransform`](#jax.experimental.pallas.mosaic_gpu.TilingTransform)
  - [`TilingTransform.__init__()`](#jax.experimental.pallas.mosaic_gpu.TilingTransform.__init__)

# jax.experimental.pallas.mosaic_gpu.TilingTransform[\#](#jax-experimental-pallas-mosaic-gpu-tilingtransform "Link to this heading")

*class* jax.experimental.pallas.mosaic_gpu.TilingTransform(*tiling*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic_gpu/core.py#L625-L654)[\#](#jax.experimental.pallas.mosaic_gpu.TilingTransform "Link to this definition")  
Represents a tiling transformation for memory refs.

A tiling of (X, Y) on an array of shape (M, N) will result in a transformed shape of (M // X, N // Y, X, Y). Ex. A (256, 256) block that is tiled with a tiling of (64, 32) will be tiled as (4, 8, 64, 32).

Parameters:  
**tiling** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *...\]*)

\_\_init\_\_(*tiling*)[\#](#jax.experimental.pallas.mosaic_gpu.TilingTransform.__init__ "Link to this definition")  
Parameters:  
**tiling** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *...\]*)

Return type:  
None

Methods

|  |  |
|----|----|
| [`__init__`](#jax.experimental.pallas.mosaic_gpu.TilingTransform.__init__ "jax.experimental.pallas.mosaic_gpu.TilingTransform.__init__")(tiling) |  |
| `pretty_print`(context) |  |
| `transform_type`(x) |  |
| `undo`(x) |  |

Attributes

|          |     |
|----------|-----|
| `tiling` |     |

[](jax.experimental.pallas.mosaic_gpu.SwizzleTransform.html "previous page")

previous

jax.experimental.pallas.mosaic_gpu.SwizzleTransform

[](jax.experimental.pallas.mosaic_gpu.WGMMAAccumulatorRef.html "next page")

next

jax.experimental.pallas.mosaic_gpu.WGMMAAccumulatorRef

Contents

- [`TilingTransform`](#jax.experimental.pallas.mosaic_gpu.TilingTransform)
  - [`TilingTransform.__init__()`](#jax.experimental.pallas.mosaic_gpu.TilingTransform.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
