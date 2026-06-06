- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.tpu` module](../jax.experimental.pallas.tpu.html)
- jax.experimental.pallas.tpu.async_remote_copy

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.tpu.async_remote_copy.rst "Download source file")
-  .pdf

# jax.experimental.pallas.tpu.async_remote_copy

## Contents

- [`async_remote_copy()`](#jax.experimental.pallas.tpu.async_remote_copy)

# jax.experimental.pallas.tpu.async_remote_copy[\#](#jax-experimental-pallas-tpu-async-remote-copy "Link to this heading")

jax.experimental.pallas.tpu.async_remote_copy(*src_ref*, *dst_ref*, *send_sem*, *recv_sem*, *device_id*, *device_id_type=DeviceIdType.MESH*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic/primitives.py#L750-L763)[\#](#jax.experimental.pallas.tpu.async_remote_copy "Link to this definition")  
Issues a remote DMA copying from src_ref to dst_ref.

Parameters:  
**device_id_type** (*primitives.DeviceIdType*)

Return type:  
AsyncCopyDescriptor

[](jax.experimental.pallas.tpu.async_copy.html "previous page")

previous

jax.experimental.pallas.tpu.async_copy

[](jax.experimental.pallas.tpu.make_async_copy.html "next page")

next

jax.experimental.pallas.tpu.make_async_copy

Contents

- [`async_remote_copy()`](#jax.experimental.pallas.tpu.async_remote_copy)

By The JAX authors

© Copyright 2024, The JAX Authors.\
