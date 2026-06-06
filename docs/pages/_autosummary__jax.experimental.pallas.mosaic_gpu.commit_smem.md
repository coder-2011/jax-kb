- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.mosaic_gpu` module](../jax.experimental.pallas.mosaic_gpu.html)
- jax.experimental.pallas.mosaic_gpu.commit_smem

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.mosaic_gpu.commit_smem.rst "Download source file")
-  .pdf

# jax.experimental.pallas.mosaic_gpu.commit_smem

## Contents

- [`commit_smem()`](#jax.experimental.pallas.mosaic_gpu.commit_smem)

# jax.experimental.pallas.mosaic_gpu.commit_smem[\#](#jax-experimental-pallas-mosaic-gpu-commit-smem "Link to this heading")

jax.experimental.pallas.mosaic_gpu.commit_smem()[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic_gpu/primitives.py#L2879-L2882)[\#](#jax.experimental.pallas.mosaic_gpu.commit_smem "Link to this definition")  
Commits all reads from/writes to SMEM, making them visible to TMA and MMA operations.

[](jax.experimental.pallas.mosaic_gpu.SemaphoreSignal.html "previous page")

previous

jax.experimental.pallas.mosaic_gpu.SemaphoreSignal

[](jax.experimental.pallas.mosaic_gpu.copy_gmem_to_smem.html "next page")

next

jax.experimental.pallas.mosaic_gpu.copy_gmem_to_smem

Contents

- [`commit_smem()`](#jax.experimental.pallas.mosaic_gpu.commit_smem)

By The JAX authors

© Copyright 2024, The JAX Authors.\
