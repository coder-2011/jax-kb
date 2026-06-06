- [](index.html)
- [API Reference](jax.html)
- `jax.profiler` module

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](_sources/jax.profiler.rst "Download source file")
-  .pdf

# jax.profiler module

## Contents

- [Tracing and time profiling](#tracing-and-time-profiling)
- [Device memory profiling](#device-memory-profiling)

# `jax.profiler` module[\#](#module-jax.profiler "Link to this heading")

## Tracing and time profiling[\#](#tracing-and-time-profiling "Link to this heading")

[Profiling computation](profiling.html) describes how to make use of JAX’s tracing and time profiling features.

|  |  |
|----|----|
| [`start_server`](_autosummary/jax.profiler.start_server.html#jax.profiler.start_server "jax.profiler.start_server")(port\[, requires_backend\]) | Starts the profiler server on port port. |
| [`start_trace`](_autosummary/jax.profiler.start_trace.html#jax.profiler.start_trace "jax.profiler.start_trace")(log_dir\[, create_perfetto_link, ...\]) | Starts a profiler trace. |
| [`stop_trace`](_autosummary/jax.profiler.stop_trace.html#jax.profiler.stop_trace "jax.profiler.stop_trace")() | Stops the currently-running profiler trace. |
| [`trace`](_autosummary/jax.profiler.trace.html#jax.profiler.trace "jax.profiler.trace")(log_dir\[, create_perfetto_link, ...\]) | Context manager to take a profiler trace. |
| [`annotate_function`](_autosummary/jax.profiler.annotate_function.html#jax.profiler.annotate_function "jax.profiler.annotate_function")(func\[, name\]) | Decorator that generates a trace event for the execution of a function. |
| [`TraceAnnotation`](_autosummary/jax.profiler.TraceAnnotation.html#jax.profiler.TraceAnnotation "jax.profiler.TraceAnnotation")(\*args, \*\*kwargs) | Context manager that generates a trace event in the profiler. |
| [`StepTraceAnnotation`](_autosummary/jax.profiler.StepTraceAnnotation.html#jax.profiler.StepTraceAnnotation "jax.profiler.StepTraceAnnotation")(name, \*\*kwargs) | Context manager that generates a step trace event in the profiler. |
| [`register_subprocess`](_autosummary/jax.profiler.register_subprocess.html#jax.profiler.register_subprocess "jax.profiler.register_subprocess")(pid, port) | Registers a subprocess's profiler server to be profiled alongside the current process. |

## Device memory profiling[\#](#device-memory-profiling "Link to this heading")

See [Profiling device memory](device_memory_profiling.html) for an introduction to JAX’s device memory profiling features.

|  |  |
|----|----|
| [`device_memory_profile`](_autosummary/jax.profiler.device_memory_profile.html#jax.profiler.device_memory_profile "jax.profiler.device_memory_profile")(\[backend\]) | Captures a JAX device memory profile as `pprof`-format protocol buffer. |
| [`save_device_memory_profile`](_autosummary/jax.profiler.save_device_memory_profile.html#jax.profiler.save_device_memory_profile "jax.profiler.save_device_memory_profile")(filename\[, backend\]) | Collects a device memory profile and writes it to a file. |

[](_autosummary/jax.ops.segment_sum.html "previous page")

previous

jax.ops.segment_sum

[](_autosummary/jax.profiler.start_server.html "next page")

next

jax.profiler.start_server

Contents

- [Tracing and time profiling](#tracing-and-time-profiling)
- [Device memory profiling](#device-memory-profiling)

By The JAX authors

© Copyright 2024, The JAX Authors.\
