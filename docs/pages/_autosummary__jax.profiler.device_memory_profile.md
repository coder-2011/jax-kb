- [](../index.html)
- [API Reference](../jax.html)
- [`jax.profiler` module](../jax.profiler.html)
- jax.profiler.device_memory_profile

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.profiler.device_memory_profile.rst "Download source file")
-  .pdf

# jax.profiler.device_memory_profile

## Contents

- [`device_memory_profile()`](#jax.profiler.device_memory_profile)

# jax.profiler.device_memory_profile[\#](#jax-profiler-device-memory-profile "Link to this heading")

jax.profiler.device_memory_profile(*backend=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/profiler.py#L424-L451)[\#](#jax.profiler.device_memory_profile "Link to this definition")  
Captures a JAX device memory profile as `pprof`-format protocol buffer.

A device memory profile is a snapshot of the state of memory, that describes the JAX [`Array`](jax.Array.html#jax.Array "jax.Array") and executable objects present in memory and their allocation sites.

For more information how to use the device memory profiler, see [Profiling device memory](../device_memory_profiling.html).

The profiling system works by instrumenting JAX on-device allocations, capturing a Python stack trace for each allocation. The instrumentation is always enabled; [`device_memory_profile()`](#jax.profiler.device_memory_profile "jax.profiler.device_memory_profile") provides an API to capture it.

The output of [`device_memory_profile()`](#jax.profiler.device_memory_profile "jax.profiler.device_memory_profile") is a binary protocol buffer that can be interpreted and visualized by the [pprof tool](https://github.com/google/pprof).

Parameters:  
**backend** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – optional; the name of the JAX backend for which the device memory profile should be collected.

Returns:  
A byte string containing a binary pprof-format protocol buffer.

Return type:  
[bytes](https://docs.python.org/3/library/stdtypes.html#bytes "(in Python v3.14)")

[](jax.profiler.register_subprocess.html "previous page")

previous

jax.profiler.register_subprocess

[](jax.profiler.save_device_memory_profile.html "next page")

next

jax.profiler.save_device_memory_profile

Contents

- [`device_memory_profile()`](#jax.profiler.device_memory_profile)

By The JAX authors

© Copyright 2024, The JAX Authors.\
