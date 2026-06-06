- [](../index.html)
- [API Reference](../jax.html)
- jax.process_indices

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.process_indices.rst "Download source file")
-  .pdf

# jax.process_indices

## Contents

- [`process_indices()`](#jax.process_indices)

# jax.process_indices[\#](#jax-process-indices "Link to this heading")

jax.process_indices(*backend=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/xla_bridge.py#L1157-L1171)[\#](#jax.process_indices "Link to this definition")  
Returns the list of all JAX process indices associated with the backend.

Parameters:  
**backend** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *xla_client.Client* *\|* *None*) – This is an experimental feature and the API is likely to change. Optional, a string representing the xla backend: `'cpu'`, `'gpu'`, or `'tpu'`.

Returns:  
List of integer process indices.

Return type:  
[list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")\]

[](jax.process_count.html "previous page")

previous

jax.process_count

[](jax.custom_jvp.html "next page")

next

jax.custom_jvp

Contents

- [`process_indices()`](#jax.process_indices)

By The JAX authors

© Copyright 2024, The JAX Authors.\
