- [](../index.html)
- [API Reference](../jax.html)
- [`jax.tree_util` module](../jax.tree_util.html)
- jax.tree_util.Partial

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.tree_util.Partial.rst "Download source file")
-  .pdf

# jax.tree_util.Partial

## Contents

- [`Partial`](#jax.tree_util.Partial)
  - [`Partial.__init__()`](#jax.tree_util.Partial.__init__)

# jax.tree_util.Partial[\#](#jax-tree-util-partial "Link to this heading")

*class* jax.tree_util.Partial(*func*, *\*args*, *\*\*kw*)[\#](#jax.tree_util.Partial "Link to this definition")  
A version of functools.partial that works in pytrees.

Use it for partial function evaluation in a way that is compatible with JAX’s transformations, e.g., `Partial(func,`` ``*args,`` ``**kwargs)`.

(You need to explicitly opt-in to this behavior because we didn’t want to give functools.partial different semantics than normal function closures.)

For example, here is a basic usage of `Partial` in a manner similar to `functools.partial`:

    >>> import jax.numpy as jnp
    >>> add_one = Partial(jnp.add, 1)
    >>> add_one(2)
    Array(3, dtype=int32, weak_type=True)

Pytree compatibility means that the resulting partial function can be passed as an argument within transformed JAX functions, which is not possible with a standard `functools.partial` function:

    >>> from jax import jit
    >>> @jit
    ... def call_func(f, *args):
    ...   return f(*args)
    ...
    >>> call_func(add_one, 2)
    Array(3, dtype=int32, weak_type=True)

Passing zero arguments to `Partial` effectively wraps the original function, making it a valid argument in JAX transformed functions:

    >>> call_func(Partial(jnp.add), 1, 2)
    Array(3, dtype=int32, weak_type=True)

Had we passed `jnp.add` to `call_func` directly, it would have resulted in a `TypeError`.

Note that if the result of `Partial` is used in the context where the value is traced, it results in all bound arguments being traced when passed to the partially-evaluated function:

    >>> print_zero = Partial(print, 0)
    >>> print_zero()
    0
    >>> call_func(print_zero)  
    JitTracer(~int32[])

\_\_init\_\_()[\#](#jax.tree_util.Partial.__init__ "Link to this definition")  

Methods

|  |  |
|----|----|
| [`__init__`](#jax.tree_util.Partial.__init__ "jax.tree_util.Partial.__init__")() |  |

Attributes

|            |                                                         |
|------------|---------------------------------------------------------|
| `args`     | tuple of arguments to future partial calls              |
| `func`     | function object to use in future partial calls          |
| `keywords` | dictionary of keyword arguments to future partial calls |

[](../jax.tree_util.html "previous page")

previous

`jax.tree_util` module

[](jax.tree_util.all_leaves.html "next page")

next

jax.tree_util.all_leaves

Contents

- [`Partial`](#jax.tree_util.Partial)
  - [`Partial.__init__()`](#jax.tree_util.Partial.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
