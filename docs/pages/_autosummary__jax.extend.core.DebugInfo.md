- [](../index.html)
- [API Reference](../jax.html)
- [`jax.extend` module](../jax.extend.html)
- [`jax.extend.core` module](../jax.extend.core.html)
- jax.extend.core.DebugInfo

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.extend.core.DebugInfo.rst "Download source file")
-  .pdf

# jax.extend.core.DebugInfo

## Contents

- [`DebugInfo`](#jax.extend.core.DebugInfo)
  - [`DebugInfo.__init__()`](#jax.extend.core.DebugInfo.__init__)

# jax.extend.core.DebugInfo[\#](#jax-extend-core-debuginfo "Link to this heading")

*class* jax.extend.core.DebugInfo(*traced_for*, *func_src_info*, *arg_names*, *result_paths*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/linear_util.py#L284-L395)[\#](#jax.extend.core.DebugInfo "Link to this definition")  
Debugging info about a func, its arguments, and results.

Parameters:  
- **traced_for** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)"))

- **func_src_info** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)"))

- **arg_names** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*,* *...\]* *\|* *None*)

- **result_paths** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*,* *...\]* *\|* *InitialResultPaths* *\|* [*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[\[\],* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*,* *...\]\]* *\|* *None*)

\_\_init\_\_()[\#](#jax.extend.core.DebugInfo.__init__ "Link to this definition")  

Methods

|  |  |
|----|----|
| [`__init__`](#jax.extend.core.DebugInfo.__init__ "jax.extend.core.DebugInfo.__init__")() |  |
| `assert_arg_names`(expected_count) |  |
| `assert_result_paths`(expected_count) |  |
| `count`(value, /) | Return number of occurrences of value. |
| `filter_arg_names`(keep) | Keep only the arg_names for which keep is True. |
| `filter_result_paths`(keep) | Keep only the result_paths for which keep is True. |
| `index`(value\[, start, stop\]) | Return first index of value. |
| `replace_func_name`(name) |  |
| `resolve_result_paths`() | Return a debug info with resolved result paths. |
| `safe_arg_names`(expected_count) | Get the arg_names with a safety check. |
| `safe_result_paths`(expected_count) | Get the result paths with a safety check. |
| `set_result_paths`(ans) |  |
| `with_unknown_names`() |  |

Attributes

|  |  |
|----|----|
| `arg_names` | The paths of the flattened non-static argnames, for example `('x',`` ``'dict_arg["a"]',`` ``...)`. |
| `func_filename` |  |
| `func_lineno` |  |
| `func_name` |  |
| `func_src_info` | e.g. `f'{fun.__name__}`` ``at`` ``{filename}:{lineno}'` or `'{fun.__name__}'` if we have no source location information. |
| `result_paths` | The paths to the flattened results, e.g., ('result\[0\]', result\[1\]) for a function that returns a tuple of arrays, or (result,) for a function that returns a single array. |
| `traced_for` | Alias for field number 0 |

[](jax.extend.core.ClosedJaxpr.html "previous page")

previous

jax.extend.core.ClosedJaxpr

[](jax.extend.core.DropVar.html "next page")

next

jax.extend.core.DropVar

Contents

- [`DebugInfo`](#jax.extend.core.DebugInfo)
  - [`DebugInfo.__init__()`](#jax.extend.core.DebugInfo.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
