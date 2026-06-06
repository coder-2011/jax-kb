- [](../index.html)
- [API Reference](../jax.html)
- [`jax.profiler` module](../jax.profiler.html)
- jax.profiler.save_device_memory_profile

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.profiler.save_device_memory_profile.rst "Download source file")
-  .pdf

# jax.profiler.save_device_memory_profile

## Contents

- [`save_device_memory_profile()`](#jax.profiler.save_device_memory_profile)

# jax.profiler.save_device_memory_profile[\#](#jax-profiler-save-device-memory-profile "Link to this heading")

jax.profiler.save_device_memory_profile(*filename*, *backend=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/profiler.py#L453-L468)[\#](#jax.profiler.save_device_memory_profile "Link to this definition")  
Collects a device memory profile and writes it to a file.

[`save_device_memory_profile()`](#jax.profiler.save_device_memory_profile "jax.profiler.save_device_memory_profile") is a convenience wrapper around [`device_memory_profile()`](jax.profiler.device_memory_profile.html#jax.profiler.device_memory_profile "jax.profiler.device_memory_profile") that saves its output to a `filename`. See the [`device_memory_profile()`](jax.profiler.device_memory_profile.html#jax.profiler.device_memory_profile "jax.profiler.device_memory_profile") documentation for more information.

Parameters:  
- **filename** – the filename to which the profile should be written.

- **backend** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – optional; the name of the JAX backend for which the device memory profile should be collected.

Return type:  
None

[](jax.profiler.device_memory_profile.html "previous page")

previous

jax.profiler.device_memory_profile

[](../jax.ref.html "next page")

next

`jax.ref` module

Contents

- [`save_device_memory_profile()`](#jax.profiler.save_device_memory_profile)

By The JAX authors

© Copyright 2024, The JAX Authors.\
