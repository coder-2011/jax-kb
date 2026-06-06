- [](../index.html)
- [API Reference](../jax.html)
- [`jax.extend` module](../jax.extend.html)
- [`jax.extend.linear_util` module](../jax.extend.linear_util.html)
- jax.extend.linear_util.WrappedFun

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.extend.linear_util.WrappedFun.rst "Download source file")
-  .pdf

# jax.extend.linear_util.WrappedFun

## Contents

- [`WrappedFun`](#jax.extend.linear_util.WrappedFun)
  - [`WrappedFun.__init__()`](#jax.extend.linear_util.WrappedFun.__init__)

# jax.extend.linear_util.WrappedFun[\#](#jax-extend-linear-util-wrappedfun "Link to this heading")

*class* jax.extend.linear_util.WrappedFun(*f*, *f_transformed*, *transforms*, *stores*, *params*, *in_type*, *debug_info*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/linear_util.py#L145-L237)[\#](#jax.extend.linear_util.WrappedFun "Link to this definition")  
Represents a function f to which transforms are to be applied.

Parameters:  
- **f** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")) – the function to be transformed.

- **f_transformed** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")) – transformed function.

- **transforms** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*,* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[Hashable,* *...\]\],* *...\]*) – a tuple of (gen, gen_static_args) tuples representing transformations to apply to f. Here gen is a generator function and gen_static_args is a tuple of static arguments for the generator. See description at the start of this module for the expected behavior of the generator.

- **stores** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[Store* *\|* *EqualStore* *\|* *None,* *...\]*) – a list of out_store for the auxiliary output of the transforms.

- **params** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*,* *Any\],* *...\]*) – a tuple of (name, param) tuples representing extra parameters to pass as keyword arguments to f, along with the transformed keyword arguments.

- **in_type** (*core.InputType* *\|* *None*) – optional input type

- **debug_info** ([*DebugInfo*](jax.extend.core.DebugInfo.html#jax.extend.core.DebugInfo "jax.extend.core.DebugInfo")) – debugging info about the function being wrapped.

\_\_init\_\_(*f*, *f_transformed*, *transforms*, *stores*, *params*, *in_type*, *debug_info*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/linear_util.py#L173-L187)[\#](#jax.extend.linear_util.WrappedFun.__init__ "Link to this definition")  
Parameters:  
- **f** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable"))

- **f_transformed** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable"))

- **transforms** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*,* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[Hashable,* *...\]\],* *...\]*)

- **stores** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[Store* *\|* *EqualStore* *\|* *None,* *...\]*)

- **params** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*,* *Hashable\],* *...\]*)

- **in_type** (*core.InputType* *\|* *None*)

- **debug_info** ([*DebugInfo*](jax.extend.core.DebugInfo.html#jax.extend.core.DebugInfo "jax.extend.core.DebugInfo"))

Methods

|  |  |
|----|----|
| [`__init__`](#jax.extend.linear_util.WrappedFun.__init__ "jax.extend.linear_util.WrappedFun.__init__")(f, f_transformed, transforms, ...) |  |
| `call_wrapped`(\*args, \*\*kwargs) | Calls the transformed function |
| `populate_stores`(stores) | Copy the values from the stores into self.stores. |
| `replace_debug_info`(dbg) |  |
| `with_unknown_names`() |  |
| `wrap`(gen, gen_static_args, out_store) | Add another transform and its store. |

Attributes

|                 |     |
|-----------------|-----|
| `f`             |     |
| `f_transformed` |     |
| `transforms`    |     |
| `stores`        |     |
| `params`        |     |
| `in_type`       |     |
| `debug_info`    |     |

[](jax.extend.linear_util.StoreException.html "previous page")

previous

jax.extend.linear_util.StoreException

[](jax.extend.linear_util.cache.html "next page")

next

jax.extend.linear_util.cache

Contents

- [`WrappedFun`](#jax.extend.linear_util.WrappedFun)
  - [`WrappedFun.__init__()`](#jax.extend.linear_util.WrappedFun.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
