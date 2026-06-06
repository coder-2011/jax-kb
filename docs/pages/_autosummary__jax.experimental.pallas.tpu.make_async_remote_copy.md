- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.tpu` module](../jax.experimental.pallas.tpu.html)
- jax.experimental.pallas.tpu.make_async_remote_copy

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.tpu.make_async_remote_copy.rst "Download source file")
-  .pdf

# jax.experimental.pallas.tpu.make_async_remote_copy

## Contents

- [`make_async_remote_copy()`](#jax.experimental.pallas.tpu.make_async_remote_copy)

# jax.experimental.pallas.tpu.make_async_remote_copy[\#](#jax-experimental-pallas-tpu-make-async-remote-copy "Link to this heading")

jax.experimental.pallas.tpu.make_async_remote_copy(*src_ref*, *dst_ref*, *send_sem*, *recv_sem*, *device_id*, *device_id_type=DeviceIdType.MESH*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/mosaic/primitives.py#L707-L748)[\#](#jax.experimental.pallas.tpu.make_async_remote_copy "Link to this definition")  
Creates a description of a remote copy operation.

Copies data from src_ref on the current device to dst_ref on the device specified by device_id. Both semaphores should be waited on using the descriptor on both source and target devices.

Note that device_id can also refer to the current device.

Parameters:  
- **src_ref** – The source Reference.

- **dst_ref** – The destination Reference.

- **send_sem** – The semaphore on the source device.

- **recv_sem** – The semaphore on the destination device.

- **device_id** (*MultiDimDeviceId* *\|* *IntDeviceId* *\|* *None*) – The device id of the destination device. It could be a tuple, or a dictionary specifying the communication axis and destination index.

- **device_id_type** (*primitives.DeviceIdType*) – The type of the device id.

Returns:  
An AsyncCopyDescriptor.

Return type:  
AsyncCopyDescriptor

[](jax.experimental.pallas.tpu.make_async_copy.html "previous page")

previous

jax.experimental.pallas.tpu.make_async_copy

[](jax.experimental.pallas.tpu.sync_copy.html "next page")

next

jax.experimental.pallas.tpu.sync_copy

Contents

- [`make_async_remote_copy()`](#jax.experimental.pallas.tpu.make_async_remote_copy)

By The JAX authors

© Copyright 2024, The JAX Authors.\
