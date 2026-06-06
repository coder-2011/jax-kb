- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.tpu` module](../jax.experimental.pallas.tpu.html)
- jax.experimental.pallas.tpu.get_tpu_info

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.tpu.get_tpu_info.rst "Download source file")
-  .pdf

# jax.experimental.pallas.tpu.get_tpu_info

## Contents

- [`get_tpu_info()`](#jax.experimental.pallas.tpu.get_tpu_info)

# jax.experimental.pallas.tpu.get_tpu_info[\#](#jax-experimental-pallas-tpu-get-tpu-info "Link to this heading")

jax.experimental.pallas.tpu.get_tpu_info()[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic/tpu_info.py#L503-L517)[\#](#jax.experimental.pallas.tpu.get_tpu_info "Link to this definition")  
Returns the TPU hardware info for the current device.

Note that all information is *per-TensorCore* so you would need to multiply by num_cores to obtain the total for the chip.

Return type:  
[*TpuInfo*](jax.experimental.pallas.tpu.TpuInfo.html#jax.experimental.pallas.tpu.TpuInfo "jax._src.pallas.mosaic.tpu_info.TpuInfo")

[](jax.experimental.pallas.tpu.get_barrier_semaphore.html "previous page")

previous

jax.experimental.pallas.tpu.get_barrier_semaphore

[](jax.experimental.pallas.tpu.is_tpu_device.html "next page")

next

jax.experimental.pallas.tpu.is_tpu_device

Contents

- [`get_tpu_info()`](#jax.experimental.pallas.tpu.get_tpu_info)

By The JAX authors

© Copyright 2024, The JAX Authors.\
