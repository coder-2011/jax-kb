- [](../index.html)
- [API Reference](../jax.html)
- [`jax.debug` module](../jax.debug.html)
- jax.debug.callback

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.debug.callback.rst "Download source file")
-  .pdf

# jax.debug.callback

## Contents

- [`callback()`](#jax.debug.callback)

# jax.debug.callback[\#](#jax-debug-callback "Link to this heading")

jax.debug.callback(*callback=None*, *\*args*, *ordered=False*, *partitioned=False*, *\*\*kwargs*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/debugging.py#L426-L526)[\#](#jax.debug.callback "Link to this definition")  
Calls a stageable Python callback.

For more explanation, see [External Callbacks](https://docs.jax.dev/en/latest/notebooks/external_callbacks.html).

`jax.debug.callback` enables you to pass in a Python function that can be called inside of a staged JAX program. A `jax.debug.callback` follows existing JAX transformation *pure* operational semantics, which are therefore unaware of side-effects. This means the effect could be dropped, duplicated, or potentially reordered in the presence of higher-order primitives and transformations.

We want this behavior because we’d like `jax.debug.callback` to be “innocuous”, i.e. we want these primitives to change the JAX computation as little as possible while revealing as much about them as possible, such as which parts of the computation are duplicated or dropped.

`jax.debug.callback` supports two ways of being called:

1.  Two-call form (Recommended): `jax.debug.callback(ordered=True)(callback,`` ``*args,`` ``**kwargs)` Options are passed in the first call. The callback and its arguments are passed in the second call. No option arguments are accepted in the second call.

2.  Single-call form: `jax.debug.callback(callback,`` ``*args,`` ``ordered=True,`` ``**kwargs)` (Soft deprecated) Mixing ordered and partitioned options with callback `kwargs` is soft deprecated.

Parameters:  
- **callback** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[...,* *None\]* *\|* *None*) – A Python callable returning None.

- **\*args** (*Any*) – The positional arguments to the callback.

- **ordered** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – A keyword only argument used to indicate whether or not the staged out computation will enforce ordering of this callback w.r.t. other ordered callbacks.

- **partitioned** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – If True, then print local shards only; this option avoids an all-gather of the operands. If False, print with logical operands; this option requires an all-gather of operands first.

- **\*\*kwargs** (*Any*) – The keyword arguments to the callback.

Returns:  
None

Return type:  
[Callable](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")\[…, None\] \| None

See also

- [`jax.experimental.io_callback()`](jax.experimental.io_callback.html#jax.experimental.io_callback "jax.experimental.io_callback"): callback designed for impure functions.

- [`jax.pure_callback()`](jax.pure_callback.html#jax.pure_callback "jax.pure_callback"): callback designed for pure functions.

- [`jax.debug.print()`](jax.debug.print.html#jax.debug.print "jax.debug.print"): callback designed for printing.

[](../jax.debug.html "previous page")

previous

`jax.debug` module

[](jax.debug.print.html "next page")

next

jax.debug.print

Contents

- [`callback()`](#jax.debug.callback)

By The JAX authors

© Copyright 2024, The JAX Authors.\
