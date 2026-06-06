- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.mosaic_gpu` module](../jax.experimental.pallas.mosaic_gpu.html)
- jax.experimental.pallas.mosaic_gpu.commit_tmem

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.mosaic_gpu.commit_tmem.rst "Download source file")
-  .pdf

# jax.experimental.pallas.mosaic_gpu.commit_tmem

## Contents

- [`commit_tmem()`](#jax.experimental.pallas.mosaic_gpu.commit_tmem)

# jax.experimental.pallas.mosaic_gpu.commit_tmem[\#](#jax-experimental-pallas-mosaic-gpu-commit-tmem "Link to this heading")

jax.experimental.pallas.mosaic_gpu.commit_tmem()[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic_gpu/primitives.py#L2819-L2827)[\#](#jax.experimental.pallas.mosaic_gpu.commit_tmem "Link to this definition")  
Commits all writes to TMEM issued by the current thread.

Once this function returns, the effects of calling `async_store_tmem` from the current thread are visible to TMEM loads, MMA and barrier operations of ``` Barrier``s ```` ``with`` ```` ``orders_tensor_core=True ```.

[](jax.experimental.pallas.mosaic_gpu.wait_load_tmem.html "previous page")

previous

jax.experimental.pallas.mosaic_gpu.wait_load_tmem

[](jax.experimental.pallas.mosaic_gpu.try_cluster_cancel.html "next page")

next

jax.experimental.pallas.mosaic_gpu.try_cluster_cancel

Contents

- [`commit_tmem()`](#jax.experimental.pallas.mosaic_gpu.commit_tmem)

By The JAX authors

© Copyright 2024, The JAX Authors.\
