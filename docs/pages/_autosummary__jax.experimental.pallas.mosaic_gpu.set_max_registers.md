- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.mosaic_gpu` module](../jax.experimental.pallas.mosaic_gpu.html)
- jax.experimental.pallas.mosaic_gpu.set_max_registers

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.mosaic_gpu.set_max_registers.rst "Download source file")
-  .pdf

# jax.experimental.pallas.mosaic_gpu.set_max_registers

## Contents

- [`set_max_registers()`](#jax.experimental.pallas.mosaic_gpu.set_max_registers)

# jax.experimental.pallas.mosaic_gpu.set_max_registers[\#](#jax-experimental-pallas-mosaic-gpu-set-max-registers "Link to this heading")

jax.experimental.pallas.mosaic_gpu.set_max_registers(*n*, *\**, *action*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic_gpu/primitives.py#L2856-L2859)[\#](#jax.experimental.pallas.mosaic_gpu.set_max_registers "Link to this definition")  
Sets the maximum number of per-lane registers in the thread.

Parameters:  
- **n** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **action** ([*Literal*](jax.extend.core.Literal.html#jax.extend.core.Literal "jax.extend.core.Literal")*\['increase',* *'decrease'\]*)

[](jax.experimental.pallas.mosaic_gpu.layout_cast.html "previous page")

previous

jax.experimental.pallas.mosaic_gpu.layout_cast

[](jax.experimental.pallas.mosaic_gpu.planar_snake.html "next page")

next

jax.experimental.pallas.mosaic_gpu.planar_snake

Contents

- [`set_max_registers()`](#jax.experimental.pallas.mosaic_gpu.set_max_registers)

By The JAX authors

© Copyright 2024, The JAX Authors.\
