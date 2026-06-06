- [](../index.html)
- [API Reference](../jax.html)
- [`jax.debug` module](../jax.debug.html)
- jax.debug.print

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.debug.print.rst "Download source file")
-  .pdf

# jax.debug.print

## Contents

- [`print()`](#jax.debug.print)

# jax.debug.print[\#](#jax-debug-print "Link to this heading")

jax.debug.print(*fmt=None*, *\*args*, *ordered=False*, *partitioned=False*, *skip_format_check=False*, *\_use_logging=False*, *\*\*kwargs*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/debugging.py#L602-L685)[\#](#jax.debug.print "Link to this definition")  
Prints values and works in staged out JAX functions.

This function does *not* work with f-strings because formatting is delayed. So instead of `jax.debug.print(f"hello`` ``{bar}")`, write `jax.debug.print("hello`` ``{bar}",`` ``bar=bar)`.

`jax.debug.print` supports two ways of being called:

1.  Two-call form (Recommended): `jax.debug.print(ordered=True)("hello`` ``{x}",`` ``x=42)` Options are passed in the first call. The format string and arguments are passed in the second call. No option arguments are accepted in the second call.

2.  Single-call form: `jax.debug.print("hello`` ``{x}",`` ``x=42,`` ``ordered=True)` (Soft deprecated) Mixing ordered and partitioned options with print `kwargs` is soft deprecated.

Parameters:  
- **fmt** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – A format string, e.g. `"hello`` ``{x}"`, that will be used to format input arguments, like `str.format`. See the Python docs on [string formatting](https://docs.python.org/3/library/stdtypes.html#str.format) and [format string syntax](https://docs.python.org/3/library/string.html#formatstrings).

- **\*args** – A list of positional arguments to be formatted, as if passed to `fmt.format`.

- **ordered** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – A keyword only argument used to indicate whether or not the staged out computation will enforce ordering of this `jax.debug.print` w.r.t. other ordered `jax.debug.print` calls.

- **partitioned** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – If True, then print local shards only; this option avoids an all-gather of the operands. If False, print with logical operands; this option requires an all-gather of operands first.

- **skip_format_check** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – If True, the format string is not checked. This is useful when using the function from inside a Pallas TPU kernel, where scalars args will be printed after the format string.

- **\*\*kwargs** – Additional keyword arguments to be formatted, as if passed to `fmt.format`.

- **\_use_logging** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

Return type:  
[Callable](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")\[…, None\] \| None

[](jax.debug.callback.html "previous page")

previous

jax.debug.callback

[](jax.debug.breakpoint.html "next page")

next

jax.debug.breakpoint

Contents

- [`print()`](#jax.debug.print)

By The JAX authors

© Copyright 2024, The JAX Authors.\
