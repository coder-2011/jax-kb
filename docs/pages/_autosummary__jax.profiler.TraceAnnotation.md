- [](../index.html)
- [API Reference](../jax.html)
- [`jax.profiler` module](../jax.profiler.html)
- jax.profiler.TraceAnnotation

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.profiler.TraceAnnotation.rst "Download source file")
-  .pdf

# jax.profiler.TraceAnnotation

## Contents

- [`TraceAnnotation`](#jax.profiler.TraceAnnotation)
  - [`TraceAnnotation.__init__()`](#jax.profiler.TraceAnnotation.__init__)

# jax.profiler.TraceAnnotation[\#](#jax-profiler-traceannotation "Link to this heading")

*class* jax.profiler.TraceAnnotation(*\*args*, *\*\*kwargs*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/profiler.py#L347-L361)[\#](#jax.profiler.TraceAnnotation "Link to this definition")  
Context manager that generates a trace event in the profiler.

The trace event spans the duration of the code enclosed by the context.

For example:

    >>> x = jnp.ones((1000, 1000))
    >>> with jax.profiler.TraceAnnotation("my_label"):
    ...   result = jnp.dot(x, x.T).block_until_ready()

This will cause a “my_label” event to show up on the trace timeline if the event occurs while the process is being traced.

\_\_init\_\_(*self*, *arg0: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*, */*, *\*\*kwargs*) → [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)")[\#](#jax.profiler.TraceAnnotation.__init__ "Link to this definition")  

Attributes

|                |     |
|----------------|-----|
| `is_enabled`   |     |
| `set_metadata` |     |

[](jax.profiler.annotate_function.html "previous page")

previous

jax.profiler.annotate_function

[](jax.profiler.StepTraceAnnotation.html "next page")

next

jax.profiler.StepTraceAnnotation

Contents

- [`TraceAnnotation`](#jax.profiler.TraceAnnotation)
  - [`TraceAnnotation.__init__()`](#jax.profiler.TraceAnnotation.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
