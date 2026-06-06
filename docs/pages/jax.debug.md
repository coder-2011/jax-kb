- [](index.html)
- [API Reference](jax.html)
- `jax.debug` module

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](_sources/jax.debug.rst "Download source file")
-  .pdf

# jax.debug module

## Contents

- [Runtime value debugging utilities](#runtime-value-debugging-utilities)
- [Sharding debugging utilities](#sharding-debugging-utilities)

# `jax.debug` module[\#](#module-jax.debug "Link to this heading")

## Runtime value debugging utilities[\#](#runtime-value-debugging-utilities "Link to this heading")

[Compiled prints and breakpoints](debugging/print_breakpoint.html) describes how to make use of JAX’s runtime value debugging features.

|  |  |
|----|----|
| [`callback`](_autosummary/jax.debug.callback.html#jax.debug.callback "jax.debug.callback")(\[callback, ordered, partitioned\]) | Calls a stageable Python callback. |
| [`print`](_autosummary/jax.debug.print.html#jax.debug.print "jax.debug.print")(\[fmt, ordered, partitioned, ...\]) | Prints values and works in staged out JAX functions. |
| [`breakpoint`](_autosummary/jax.debug.breakpoint.html#jax.debug.breakpoint "jax.debug.breakpoint")(\*\[, backend, filter_frames, ...\]) | Enters a breakpoint at a point in a program. |

## Sharding debugging utilities[\#](#sharding-debugging-utilities "Link to this heading")

Functions that enable inspecting and visualizing array shardings inside (and outside) staged functions.

|  |  |
|----|----|
| [`inspect_array_sharding`](_autosummary/jax.debug.inspect_array_sharding.html#jax.debug.inspect_array_sharding "jax.debug.inspect_array_sharding")(value, \*, callback) | Enables inspecting array sharding inside JIT-ted functions. |
| [`visualize_array_sharding`](_autosummary/jax.debug.visualize_array_sharding.html#jax.debug.visualize_array_sharding "jax.debug.visualize_array_sharding")(arr, \*\*kwargs) | Visualizes an array's sharding. |
| [`visualize_sharding`](_autosummary/jax.debug.visualize_sharding.html#jax.debug.visualize_sharding "jax.debug.visualize_sharding")(shape, sharding, \*\[, ...\]) | Visualizes a `Sharding` using `rich`. |

[](_autosummary/jax.ad_checkpoint.checkpoint_name.html "previous page")

previous

jax.ad_checkpoint.checkpoint_name

[](_autosummary/jax.debug.callback.html "next page")

next

jax.debug.callback

Contents

- [Runtime value debugging utilities](#runtime-value-debugging-utilities)
- [Sharding debugging utilities](#sharding-debugging-utilities)

By The JAX authors

© Copyright 2024, The JAX Authors.\
