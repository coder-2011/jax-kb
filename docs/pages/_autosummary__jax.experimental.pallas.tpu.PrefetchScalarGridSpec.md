- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.tpu` module](../jax.experimental.pallas.tpu.html)
- jax.experimental.pallas.tpu.PrefetchScalarGridSpec

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.tpu.PrefetchScalarGridSpec.rst "Download source file")
-  .pdf

# jax.experimental.pallas.tpu.PrefetchScalarGridSpec

## Contents

- [`PrefetchScalarGridSpec`](#jax.experimental.pallas.tpu.PrefetchScalarGridSpec)
  - [`PrefetchScalarGridSpec.__init__()`](#jax.experimental.pallas.tpu.PrefetchScalarGridSpec.__init__)

# jax.experimental.pallas.tpu.PrefetchScalarGridSpec[\#](#jax-experimental-pallas-tpu-prefetchscalargridspec "Link to this heading")

*class* jax.experimental.pallas.tpu.PrefetchScalarGridSpec(*num_scalar_prefetch: 'int'*, *grid: 'pallas_core.Grid' = ()*, *in_specs: 'pallas_core.BlockSpecTree' = NoBlockSpec*, *out_specs: 'pallas_core.BlockSpecTree' = NoBlockSpec*, *scratch_shapes: 'pallas_core.ScratchShapeTree' = ()*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic/core.py#L304-L324)[\#](#jax.experimental.pallas.tpu.PrefetchScalarGridSpec "Link to this definition")  
Parameters:  
- **num_scalar_prefetch** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **grid** (*TupleGrid*)

- **in_specs** (*BlockSpecTree*)

- **out_specs** (*BlockSpecTree*)

- **scratch_shapes** (*ScratchShapeTree*)

\_\_init\_\_(*num_scalar_prefetch*, *grid=()*, *in_specs=NoBlockSpec*, *out_specs=NoBlockSpec*, *scratch_shapes=()*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic/core.py#L308-L319)[\#](#jax.experimental.pallas.tpu.PrefetchScalarGridSpec.__init__ "Link to this definition")  
Parameters:  
- **num_scalar_prefetch** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **grid** (*pallas_core.Grid*)

- **in_specs** (*pallas_core.BlockSpecTree*)

- **out_specs** (*pallas_core.BlockSpecTree*)

- **scratch_shapes** (*pallas_core.ScratchShapeTree*)

Methods

|  |  |
|----|----|
| [`__init__`](#jax.experimental.pallas.tpu.PrefetchScalarGridSpec.__init__ "jax.experimental.pallas.tpu.PrefetchScalarGridSpec.__init__")(num_scalar_prefetch\[, grid, ...\]) |  |

Attributes

|                       |     |
|-----------------------|-----|
| `scratch_shapes`      |     |
| `num_scalar_prefetch` |     |
| `grid`                |     |
| `grid_names`          |     |
| `in_specs`            |     |
| `out_specs`           |     |

[](jax.experimental.pallas.tpu.MemorySpace.html "previous page")

previous

jax.experimental.pallas.tpu.MemorySpace

[](jax.experimental.pallas.tpu.SemaphoreType.html "next page")

next

jax.experimental.pallas.tpu.SemaphoreType

Contents

- [`PrefetchScalarGridSpec`](#jax.experimental.pallas.tpu.PrefetchScalarGridSpec)
  - [`PrefetchScalarGridSpec.__init__()`](#jax.experimental.pallas.tpu.PrefetchScalarGridSpec.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
