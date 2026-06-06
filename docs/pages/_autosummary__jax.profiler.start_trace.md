- [](../index.html)
- [API Reference](../jax.html)
- [`jax.profiler` module](../jax.profiler.html)
- jax.profiler.start_trace

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.profiler.start_trace.rst "Download source file")
-  .pdf

# jax.profiler.start_trace

## Contents

- [`start_trace()`](#jax.profiler.start_trace)

# jax.profiler.start_trace[\#](#jax-profiler-start-trace "Link to this heading")

jax.profiler.start_trace(*log_dir*, *create_perfetto_link=False*, *create_perfetto_trace=False*, *profiler_options=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/profiler.py#L151-L211)[\#](#jax.profiler.start_trace "Link to this definition")  
Starts a profiler trace.

The trace will capture CPU, GPU, and/or TPU activity, including Python functions and JAX on-device operations. Use [`stop_trace()`](jax.profiler.stop_trace.html#jax.profiler.stop_trace "jax.profiler.stop_trace") to end the trace and save the results to `log_dir`.

The resulting trace can be viewed with TensorBoard. Note that TensorBoard doesn’t need to be running when collecting the trace.

Only one trace may be collected at a time. A RuntimeError will be raised if [`start_trace()`](#jax.profiler.start_trace "jax.profiler.start_trace") is called while another trace is running.

Parameters:  
- **log_dir** ([*os.PathLike*](https://docs.python.org/3/library/os.html#os.PathLike "(in Python v3.14)") *\|* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – The directory to save the profiler trace to (usually the TensorBoard log directory).

- **create_perfetto_link** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – A boolean which, if true, creates and prints link to the Perfetto trace viewer UI ([https://ui.perfetto.dev](https://ui.perfetto.dev)). The program will block until the link is opened and Perfetto loads the trace.

- **create_perfetto_trace** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – A boolean which, if true, additionally dumps a `perfetto_trace.json.gz` file that is compatible for upload with the Perfetto trace viewer UI ([https://ui.perfetto.dev](https://ui.perfetto.dev)). The file will also be generated if `create_perfetto_link` is true. This could be useful if you want to generate a Perfetto-compatible trace without blocking the process.

- **profiler_options** (*ProfileOptions* *\|* *None*) – Profiler options to configure the profiler for collection.

Return type:  
None

[](jax.profiler.start_server.html "previous page")

previous

jax.profiler.start_server

[](jax.profiler.stop_trace.html "next page")

next

jax.profiler.stop_trace

Contents

- [`start_trace()`](#jax.profiler.start_trace)

By The JAX authors

© Copyright 2024, The JAX Authors.\
