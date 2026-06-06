- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.mosaic_gpu` module](../jax.experimental.pallas.mosaic_gpu.html)
- jax.experimental.pallas.mosaic_gpu.async_store_tmem

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.mosaic_gpu.async_store_tmem.rst "Download source file")
-  .pdf

# jax.experimental.pallas.mosaic_gpu.async_store_tmem

## Contents

- [`async_store_tmem()`](#jax.experimental.pallas.mosaic_gpu.async_store_tmem)

# jax.experimental.pallas.mosaic_gpu.async_store_tmem[\#](#jax-experimental-pallas-mosaic-gpu-async-store-tmem "Link to this heading")

jax.experimental.pallas.mosaic_gpu.async_store_tmem(*ref*, *value*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic_gpu/primitives.py#L3697-L3716)[\#](#jax.experimental.pallas.mosaic_gpu.async_store_tmem "Link to this definition")  
Stores the value to TMEM.

The store is asynchronous and is not guaranteed to be visible (e.g. by reads or MMA operations) until `commit_tmem` has been called.

Parameters:  
- **ref** (*\_Ref*) – The TMEM reference to store to.

- **value** – The value to store.

[](jax.experimental.pallas.mosaic_gpu.async_load_tmem.html "previous page")

previous

jax.experimental.pallas.mosaic_gpu.async_load_tmem

[](jax.experimental.pallas.mosaic_gpu.wait_load_tmem.html "next page")

next

jax.experimental.pallas.mosaic_gpu.wait_load_tmem

Contents

- [`async_store_tmem()`](#jax.experimental.pallas.mosaic_gpu.async_store_tmem)

By The JAX authors

© Copyright 2024, The JAX Authors.\
