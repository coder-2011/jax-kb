- [](../index.html)
- [API Reference](../jax.html)
- jax.local_devices

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.local_devices.rst "Download source file")
-  .pdf

# jax.local_devices

## Contents

- [`local_devices()`](#jax.local_devices)

# jax.local_devices[\#](#jax-local-devices "Link to this heading")

jax.local_devices(*process_index=None*, *backend=None*, *host_id=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/xla_bridge.py#L1082-L1111)[\#](#jax.local_devices "Link to this definition")  
Like [`jax.devices()`](jax.devices.html#jax.devices "jax.devices"), but only returns devices local to a given process.

If `process_index` is `None`, returns devices local to this process.

Parameters:  
- **process_index** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – the integer index of the process. Process indices can be retrieved via `len(jax.process_count())`.

- **backend** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *xla_client.Client* *\|* *None*) – This is an experimental feature and the API is likely to change. Optional, a string representing the xla backend: `'cpu'`, `'gpu'`, or `'tpu'`.

- **host_id** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*)

Returns:  
List of Device subclasses.

Return type:  
[list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")\[xla_client.Device\]

[](jax.devices.html "previous page")

previous

jax.devices

[](jax.process_index.html "next page")

next

jax.process_index

Contents

- [`local_devices()`](#jax.local_devices)

By The JAX authors

© Copyright 2024, The JAX Authors.\
