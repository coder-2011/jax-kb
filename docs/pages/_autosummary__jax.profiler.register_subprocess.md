- [](../index.html)
- [API Reference](../jax.html)
- [`jax.profiler` module](../jax.profiler.html)
- jax.profiler.register_subprocess

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.profiler.register_subprocess.rst "Download source file")
-  .pdf

# jax.profiler.register_subprocess

## Contents

- [`register_subprocess()`](#jax.profiler.register_subprocess)

# jax.profiler.register_subprocess[\#](#jax-profiler-register-subprocess "Link to this heading")

jax.profiler.register_subprocess(*pid*, *port*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/profiler.py#L93-L119)[\#](#jax.profiler.register_subprocess "Link to this definition")  
Registers a subprocess’s profiler server to be profiled alongside the current process.

When the current process collects a profile (either programmatically or via its profiler server), it will propagate the request to all registered subprocesses’ profiler servers and subsequently, aggregate all their responses into the main response returned by this process’s profiler server.

This is helpful when running workers in separate processes that may affect the performance of the main process (e.g. PyGrain).

NOTE: Currently, only CPU profiling of subprocesses is supported.

Parameters:  
- **pid** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – The process ID of the subprocess.

- **port** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – The port of the profiler server in the subprocess.

Returns:  
A function that when called or garbage collected will unregister the subprocess from the main process’s profiler.

Raises:  
- [**RuntimeError**](https://docs.python.org/3/library/exceptions.html#RuntimeError "(in Python v3.14)") – If the subprocess fails to be registered (e.g. already

- **registered,** **unable to connect,** **etc.).** –

Return type:  
[Callable](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")\[\[\], None\]

[](jax.profiler.StepTraceAnnotation.html "previous page")

previous

jax.profiler.StepTraceAnnotation

[](jax.profiler.device_memory_profile.html "next page")

next

jax.profiler.device_memory_profile

Contents

- [`register_subprocess()`](#jax.profiler.register_subprocess)

By The JAX authors

© Copyright 2024, The JAX Authors.\
