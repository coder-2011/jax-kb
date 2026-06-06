- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.mosaic_gpu` module](../jax.experimental.pallas.mosaic_gpu.html)
- jax.experimental.pallas.mosaic_gpu.wgmma_wait

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.mosaic_gpu.wgmma_wait.rst "Download source file")
-  .pdf

# jax.experimental.pallas.mosaic_gpu.wgmma_wait

## Contents

- [`wgmma_wait()`](#jax.experimental.pallas.mosaic_gpu.wgmma_wait)

# jax.experimental.pallas.mosaic_gpu.wgmma_wait[\#](#jax-experimental-pallas-mosaic-gpu-wgmma-wait "Link to this heading")

jax.experimental.pallas.mosaic_gpu.wgmma_wait(*n*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic_gpu/primitives.py#L1852-L1855)[\#](#jax.experimental.pallas.mosaic_gpu.wgmma_wait "Link to this definition")  
Waits until there is no more than `n` WGMMA operations in flight.

Parameters:  
**n** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

[](jax.experimental.pallas.mosaic_gpu.wgmma.html "previous page")

previous

jax.experimental.pallas.mosaic_gpu.wgmma

[](jax.experimental.pallas.mosaic_gpu.tcgen05_mma.html "next page")

next

jax.experimental.pallas.mosaic_gpu.tcgen05_mma

Contents

- [`wgmma_wait()`](#jax.experimental.pallas.mosaic_gpu.wgmma_wait)

By The JAX authors

© Copyright 2024, The JAX Authors.\
