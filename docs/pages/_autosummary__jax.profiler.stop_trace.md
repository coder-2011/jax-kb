- [](../index.html)
- [API Reference](../jax.html)
- [`jax.profiler` module](../jax.profiler.html)
- jax.profiler.stop_trace

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.profiler.stop_trace.rst "Download source file")
-  .pdf

# jax.profiler.stop_trace

## Contents

- [`stop_trace()`](#jax.profiler.stop_trace)

# jax.profiler.stop_trace[\#](#jax-profiler-stop-trace "Link to this heading")

jax.profiler.stop_trace()[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/profiler.py#L271-L288)[\#](#jax.profiler.stop_trace "Link to this definition")  
Stops the currently-running profiler trace.

The trace will be saved to the `log_dir` passed to the corresponding [`start_trace()`](jax.profiler.start_trace.html#jax.profiler.start_trace "jax.profiler.start_trace") call. Raises a RuntimeError if a trace hasn’t been started.

[](jax.profiler.start_trace.html "previous page")

previous

jax.profiler.start_trace

[](jax.profiler.trace.html "next page")

next

jax.profiler.trace

Contents

- [`stop_trace()`](#jax.profiler.stop_trace)

By The JAX authors

© Copyright 2024, The JAX Authors.\
