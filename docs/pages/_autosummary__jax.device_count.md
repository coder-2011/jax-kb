- [](../index.html)
- [API Reference](../jax.html)
- jax.device_count

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.device_count.rst "Download source file")
-  .pdf

# jax.device_count

## Contents

- [`device_count()`](#jax.device_count)

# jax.device_count[\#](#jax-device-count "Link to this heading")

jax.device_count(*backend=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/xla_bridge.py#L985-L1005)[\#](#jax.device_count "Link to this definition")  
Returns the total number of devices.

On most platforms, this is the same as [`jax.local_device_count()`](jax.local_device_count.html#jax.local_device_count "jax.local_device_count"). However, on multi-process platforms where different devices are associated with different processes, this will return the total number of devices across all processes.

Parameters:  
**backend** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *xla_client.Client* *\|* *None*) – This is an experimental feature and the API is likely to change. Optional, a string representing the xla backend: `'cpu'`, `'gpu'`, or `'tpu'`.

Returns:  
Number of devices.

Return type:  
[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

[](jax.process_index.html "previous page")

previous

jax.process_index

[](jax.local_device_count.html "next page")

next

jax.local_device_count

Contents

- [`device_count()`](#jax.device_count)

By The JAX authors

© Copyright 2024, The JAX Authors.\
