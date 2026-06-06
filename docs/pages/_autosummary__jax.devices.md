- [](../index.html)
- [API Reference](../jax.html)
- jax.devices

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.devices.rst "Download source file")
-  .pdf

# jax.devices

## Contents

- [`devices()`](#jax.devices)

# jax.devices[\#](#jax-devices "Link to this heading")

jax.devices(*backend=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/xla_bridge.py#L1014-L1040)[\#](#jax.devices "Link to this definition")  
Returns a list of all devices for a given backend.

Each device is represented by a subclass of [`Device`](jax.Device.html#jax.Device "jaxlib._jax.Device") (e.g. `CpuDevice`, `GpuDevice`). The length of the returned list is equal to `device_count(backend)`. Local devices can be identified by comparing `Device.process_index` to the value returned by [`jax.process_index()`](jax.process_index.html#jax.process_index "jax.process_index").

If `backend` is `None`, returns all the devices from the default backend. The default backend is generally `'gpu'` or `'tpu'` if available, otherwise `'cpu'`.

Parameters:  
**backend** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *xla_client.Client* *\|* *None*) – This is an experimental feature and the API is likely to change. Optional, a string representing the xla backend: `'cpu'`, `'gpu'`, or `'tpu'`.

Returns:  
List of Device subclasses.

Return type:  
[list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")\[xla_client.Device\]

[](jax.pmap.html "previous page")

previous

jax.pmap

[](jax.local_devices.html "next page")

next

jax.local_devices

Contents

- [`devices()`](#jax.devices)

By The JAX authors

© Copyright 2024, The JAX Authors.\
