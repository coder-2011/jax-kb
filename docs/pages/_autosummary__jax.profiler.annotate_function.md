- [](../index.html)
- [API Reference](../jax.html)
- [`jax.profiler` module](../jax.profiler.html)
- jax.profiler.annotate_function

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.profiler.annotate_function.rst "Download source file")
-  .pdf

# jax.profiler.annotate_function

## Contents

- [`annotate_function()`](#jax.profiler.annotate_function)

# jax.profiler.annotate_function[\#](#jax-profiler-annotate-function "Link to this heading")

jax.profiler.annotate_function(*func*, *name=None*, *\*\*decorator_kwargs*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/profiler.py#L389-L422)[\#](#jax.profiler.annotate_function "Link to this definition")  
Decorator that generates a trace event for the execution of a function.

For example:

    >>> @jax.profiler.annotate_function
    ... def f(x):
    ...   return jnp.dot(x, x.T).block_until_ready()
    >>>
    >>> result = f(jnp.ones((1000, 1000)))

This will cause an “f” event to show up on the trace timeline if the function execution occurs while the process is being traced by TensorBoard.

Arguments can be passed to the decorator via [`functools.partial()`](https://docs.python.org/3/library/functools.html#functools.partial "(in Python v3.14)").

    >>> from functools import partial

    >>> @partial(jax.profiler.annotate_function, name="event_name")
    ... def f(x):
    ...   return jnp.dot(x, x.T).block_until_ready()

    >>> result = f(jnp.ones((1000, 1000)))

Parameters:  
- **func** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable"))

- **name** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*)

[](jax.profiler.trace.html "previous page")

previous

jax.profiler.trace

[](jax.profiler.TraceAnnotation.html "next page")

next

jax.profiler.TraceAnnotation

Contents

- [`annotate_function()`](#jax.profiler.annotate_function)

By The JAX authors

© Copyright 2024, The JAX Authors.\
