- [](../index.html)
- [API Reference](../jax.html)
- [`jax.profiler` module](../jax.profiler.html)
- jax.profiler.start_server

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.profiler.start_server.rst "Download source file")
-  .pdf

# jax.profiler.start_server

## Contents

- [`start_server()`](#jax.profiler.start_server)

# jax.profiler.start_server[\#](#jax-profiler-start-server "Link to this heading")

jax.profiler.start_server(*port*, *requires_backend=True*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/profiler.py#L53-L83)[\#](#jax.profiler.start_server "Link to this definition")  
Starts the profiler server on port port.

Using the “TensorFlow profiler” feature in [TensorBoard](https://www.tensorflow.org/tensorboard) 2.2 or newer, you can connect to the profiler server and sample execution traces that show CPU, GPU, and/or TPU device activity.

Parameters:  
- **port** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – The port to start the profiler server on.

- **requires_backend** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – If False, the profiler server will not wait for backends to be initialized before starting. Default is True.

Return type:  
\_profiler.ProfilerServer

[](../jax.profiler.html "previous page")

previous

`jax.profiler` module

[](jax.profiler.start_trace.html "next page")

next

jax.profiler.start_trace

Contents

- [`start_server()`](#jax.profiler.start_server)

By The JAX authors

© Copyright 2024, The JAX Authors.\
