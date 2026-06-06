- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.mosaic_gpu` module](../jax.experimental.pallas.mosaic_gpu.html)
- jax.experimental.pallas.mosaic_gpu.wait_load_tmem

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.mosaic_gpu.wait_load_tmem.rst "Download source file")
-  .pdf

# jax.experimental.pallas.mosaic_gpu.wait_load_tmem

## Contents

- [`wait_load_tmem()`](#jax.experimental.pallas.mosaic_gpu.wait_load_tmem)

# jax.experimental.pallas.mosaic_gpu.wait_load_tmem[\#](#jax-experimental-pallas-mosaic-gpu-wait-load-tmem "Link to this heading")

jax.experimental.pallas.mosaic_gpu.wait_load_tmem()[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic_gpu/primitives.py#L3671-L3680)[\#](#jax.experimental.pallas.mosaic_gpu.wait_load_tmem "Link to this definition")  
Awaits all previously asynchronous TMEM loads issued by the calling thread.

Once this function returns, the TMEM loads issued by the calling thread are guaranteed to have completed. The read TMEM regions can be safely overwritten by the calling thread, or any threads signalled through ``` Barrier``s ```` ``with`` ```` ``orders_tensor_core=True ```.

[](jax.experimental.pallas.mosaic_gpu.async_store_tmem.html "previous page")

previous

jax.experimental.pallas.mosaic_gpu.async_store_tmem

[](jax.experimental.pallas.mosaic_gpu.commit_tmem.html "next page")

next

jax.experimental.pallas.mosaic_gpu.commit_tmem

Contents

- [`wait_load_tmem()`](#jax.experimental.pallas.mosaic_gpu.wait_load_tmem)

By The JAX authors

© Copyright 2024, The JAX Authors.\
