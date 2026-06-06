- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.mosaic_gpu` module](../jax.experimental.pallas.mosaic_gpu.html)
- jax.experimental.pallas.mosaic_gpu.planar_snake

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.mosaic_gpu.planar_snake.rst "Download source file")
-  .pdf

# jax.experimental.pallas.mosaic_gpu.planar_snake

## Contents

- [`planar_snake()`](#jax.experimental.pallas.mosaic_gpu.planar_snake)

# jax.experimental.pallas.mosaic_gpu.planar_snake[\#](#jax-experimental-pallas-mosaic-gpu-planar-snake "Link to this heading")

jax.experimental.pallas.mosaic_gpu.planar_snake(*lin_idx*, *shape*, *minor_dim*, *tile_width*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic_gpu/helpers.py#L252-L304)[\#](#jax.experimental.pallas.mosaic_gpu.planar_snake "Link to this definition")  
Converts a linear index into an index into shape, trying to optimize locality.

The “space filling curve” this function computes splits the minor dimension into tiles of length `tile_width`. Every other tile has its major dimension inverted, so that the iteration order “snakes around” when going from one tile to another.

For a shape of (8, 8), `minor_dim=0` and `tile_width=2`, the iteration order is:

     0   2   4   6   8  10  12  14
     1   3   5   7   9  11  13  15
    30  28  26  24  22  20  18  16
    31  29  27  25  23  21  19  17
    32  34  36  38  40  42  44  46
    33  35  37  39  41  43  45  47
    62  60  58  56  54  52  50  48
    63  61  59  57  55  53  51  49

Notice how each pair of rows forms a tile (`minor_dim=0`, `tile_width=2`) and when moving from one tile to another, the indices increase along columns in one of them and decrease in the other.

Parameters:  
- **lin_idx** ([*Array*](jax.Array.html#jax.Array "jax.Array"))

- **shape** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*)

- **minor_dim** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **tile_width** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

[](jax.experimental.pallas.mosaic_gpu.set_max_registers.html "previous page")

previous

jax.experimental.pallas.mosaic_gpu.set_max_registers

[](jax.experimental.pallas.mosaic_gpu.emit_pipeline.html "next page")

next

jax.experimental.pallas.mosaic_gpu.emit_pipeline

Contents

- [`planar_snake()`](#jax.experimental.pallas.mosaic_gpu.planar_snake)

By The JAX authors

© Copyright 2024, The JAX Authors.\
