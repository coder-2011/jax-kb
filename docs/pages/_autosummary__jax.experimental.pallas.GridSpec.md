- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- jax.experimental.pallas.GridSpec

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.GridSpec.rst "Download source file")
-  .pdf

# jax.experimental.pallas.GridSpec

## Contents

- [`GridSpec`](#jax.experimental.pallas.GridSpec)
  - [`GridSpec.__init__()`](#jax.experimental.pallas.GridSpec.__init__)

# jax.experimental.pallas.GridSpec[\#](#jax-experimental-pallas-gridspec "Link to this heading")

*class* jax.experimental.pallas.GridSpec(*grid=()*, *in_specs=NoBlockSpec*, *out_specs=NoBlockSpec*, *scratch_shapes=()*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/core.py#L1193-L1243)[\#](#jax.experimental.pallas.GridSpec "Link to this definition")  
Encodes the grid parameters for [`jax.experimental.pallas.pallas_call()`](jax.experimental.pallas.pallas_call.html#jax.experimental.pallas.pallas_call "jax.experimental.pallas.pallas_call").

See the documentation for [`jax.experimental.pallas.pallas_call()`](jax.experimental.pallas.pallas_call.html#jax.experimental.pallas.pallas_call "jax.experimental.pallas.pallas_call"), and also [Grids and BlockSpecs](../pallas/grid_blockspec.html#pallas-grids-and-blockspecs) for a more detailed description of the parameters.

Parameters:  
- **grid** (*TupleGrid*)

- **in_specs** (*BlockSpecTree*)

- **out_specs** (*BlockSpecTree*)

- **scratch_shapes** (*ScratchShapeTree*)

\_\_init\_\_(*grid=()*, *in_specs=NoBlockSpec*, *out_specs=NoBlockSpec*, *scratch_shapes=()*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/core.py#L1208-L1240)[\#](#jax.experimental.pallas.GridSpec.__init__ "Link to this definition")  
Parameters:  
- **grid** (*Grid*)

- **in_specs** (*BlockSpecTree*)

- **out_specs** (*BlockSpecTree*)

- **scratch_shapes** (*ScratchShapeTree*)

Methods

|  |  |
|----|----|
| [`__init__`](#jax.experimental.pallas.GridSpec.__init__ "jax.experimental.pallas.GridSpec.__init__")(\[grid, in_specs, out_specs, ...\]) |  |

Attributes

|                  |     |
|------------------|-----|
| `scratch_shapes` |     |
| `grid`           |     |
| `grid_names`     |     |
| `in_specs`       |     |
| `out_specs`      |     |

[](jax.experimental.pallas.BlockSpec.html "previous page")

previous

jax.experimental.pallas.BlockSpec

[](jax.experimental.pallas.Slice.html "next page")

next

jax.experimental.pallas.Slice

Contents

- [`GridSpec`](#jax.experimental.pallas.GridSpec)
  - [`GridSpec.__init__()`](#jax.experimental.pallas.GridSpec.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
