- [](../index.html)
- [API Reference](../jax.html)
- [`jax.debug` module](../jax.debug.html)
- jax.debug.breakpoint

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.debug.breakpoint.rst "Download source file")
-  .pdf

# jax.debug.breakpoint

## Contents

- [`breakpoint()`](#jax.debug.breakpoint)

# jax.debug.breakpoint[\#](#jax-debug-breakpoint "Link to this heading")

jax.debug.breakpoint(*\**, *backend=None*, *filter_frames=True*, *num_frames=None*, *ordered=False*, *token=None*, *\*\*kwargs*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/debugger/core.py#L160-L231)[\#](#jax.debug.breakpoint "Link to this definition")  
Enters a breakpoint at a point in a program.

Parameters:  
- **backend** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – The debugger backend to use. By default, picks the highest priority debugger and in the absence of other registered debuggers, falls back to the CLI debugger.

- **filter_frames** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – Whether or not to filter out JAX-internal stack frames from the traceback. Since some libraries, like Flax, also make use of JAX’s stack frame filtering system, this option can also affect whether stack frames from libraries are filtered.

- **num_frames** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – The number of frames above the current stack frame to make available for inspection in the interactive debugger.

- **ordered** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – A keyword only argument used to indicate whether or not the staged out computation will enforce ordering of this `jax.debug.breakpoint` with respect to other ordered `jax.debug.breakpoint` and `jax.debug.print` calls.

- **token** – A keyword only argument; an alternative to `ordered`. If used then a JAX array (or pytree of JAX arrays) should be passed, and the breakpoint will be run once its value is computed. This is returned unchanged, and should be passed back to the computation. If the return value is unused in the later computation, then the whole computation will be pruned and this breakpoint will not be run.

Returns:  
If token is passed, then its value is returned unchanged. Otherwise, returns None.

[](jax.debug.print.html "previous page")

previous

jax.debug.print

[](jax.debug.inspect_array_sharding.html "next page")

next

jax.debug.inspect_array_sharding

Contents

- [`breakpoint()`](#jax.debug.breakpoint)

By The JAX authors

© Copyright 2024, The JAX Authors.\
