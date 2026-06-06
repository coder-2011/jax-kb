- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.mosaic_gpu` module](../jax.experimental.pallas.mosaic_gpu.html)
- jax.experimental.pallas.mosaic_gpu.wait_smem_to_gmem

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.mosaic_gpu.wait_smem_to_gmem.rst "Download source file")
-  .pdf

# jax.experimental.pallas.mosaic_gpu.wait_smem_to_gmem

## Contents

- [`wait_smem_to_gmem()`](#jax.experimental.pallas.mosaic_gpu.wait_smem_to_gmem)

# jax.experimental.pallas.mosaic_gpu.wait_smem_to_gmem[\#](#jax-experimental-pallas-mosaic-gpu-wait-smem-to-gmem "Link to this heading")

jax.experimental.pallas.mosaic_gpu.wait_smem_to_gmem(*n*, *wait_read_only=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic_gpu/primitives.py#L1453-L1462)[\#](#jax.experimental.pallas.mosaic_gpu.wait_smem_to_gmem "Link to this definition")  
Waits until no more than the most recent `n` SMEM-\>GMEM copies issued by the calling thread are in flight.

Parameters:  
- **n** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – The maximum number of copies in flight to wait for.

- **wait_read_only** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – If `True`, wait for the in flight copies to finish reading from SMEM. The writes to GMEM are not waited for.

Return type:  
None

[](jax.experimental.pallas.mosaic_gpu.copy_smem_to_gmem.html "previous page")

previous

jax.experimental.pallas.mosaic_gpu.copy_smem_to_gmem

[](jax.experimental.pallas.mosaic_gpu.wgmma.html "next page")

next

jax.experimental.pallas.mosaic_gpu.wgmma

Contents

- [`wait_smem_to_gmem()`](#jax.experimental.pallas.mosaic_gpu.wait_smem_to_gmem)

By The JAX authors

© Copyright 2024, The JAX Authors.\
