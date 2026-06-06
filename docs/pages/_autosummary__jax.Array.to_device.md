- [](../index.html)
- [API Reference](../jax.html)
- jax.Array.to_device

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.Array.to_device.rst "Download source file")
-  .pdf

# jax.Array.to_device

## Contents

- [`Array.to_device()`](#jax.Array.to_device)

# jax.Array.to_device[\#](#jax-array-to-device "Link to this heading")

*abstract* Array.to_device(*device*, *\**, *stream=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/array_methods.py#L458-L472)[\#](#jax.Array.to_device "Link to this definition")  
Return a copy of the array on the specified device

Parameters:  
- **device** (*xc.Device* *\|* [*Sharding*](../jax.sharding.html#jax.sharding.Sharding "jax.sharding.Sharding")) – [`Device`](jax.Device.html#jax.Device "jax.Device") or [`Sharding`](../jax.sharding.html#jax.sharding.Sharding "jax.sharding.Sharding") to which the created array will be committed.

- **stream** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *Any* *\|* *None*) – not implemented, passing a non-None value will lead to an error.

- **self** ([*Array*](jax.Array.html#jax.Array "jax.Array"))

Returns:  
copy of array placed on the specified device or devices.

[](jax.Array.take.html "previous page")

previous

jax.Array.take

[](jax.Array.trace.html "next page")

next

jax.Array.trace

Contents

- [`Array.to_device()`](#jax.Array.to_device)

By The JAX authors

© Copyright 2024, The JAX Authors.\
