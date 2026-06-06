- [](../index.html)
- [API Reference](../jax.html)
- jax.process_index

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.process_index.rst "Download source file")
-  .pdf

# jax.process_index

## Contents

- [`process_index()`](#jax.process_index)

# jax.process_index[\#](#jax-process-index "Link to this heading")

jax.process_index(*backend=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/xla_bridge.py#L1113-L1130)[\#](#jax.process_index "Link to this definition")  
Returns the integer process index of this process.

On most platforms, this will always be 0. This will vary on multi-process platforms though.

Parameters:  
**backend** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *xla_client.Client* *\|* *None*) – This is an experimental feature and the API is likely to change. Optional, a string representing the xla backend: `'cpu'`, `'gpu'`, or `'tpu'`.

Returns:  
Integer process index.

Return type:  
[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

[](jax.local_devices.html "previous page")

previous

jax.local_devices

[](jax.device_count.html "next page")

next

jax.device_count

Contents

- [`process_index()`](#jax.process_index)

By The JAX authors

© Copyright 2024, The JAX Authors.\
