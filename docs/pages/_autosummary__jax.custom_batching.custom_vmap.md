- [](../index.html)
- [API Reference](../jax.html)
- jax.custom_batching.custom_vmap

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.custom_batching.custom_vmap.rst "Download source file")
-  .pdf

# jax.custom_batching.custom_vmap

## Contents

- [`custom_vmap`](#jax.custom_batching.custom_vmap)
  - [`custom_vmap.__init__()`](#jax.custom_batching.custom_vmap.__init__)

# jax.custom_batching.custom_vmap[\#](#jax-custom-batching-custom-vmap "Link to this heading")

*class* jax.custom_batching.custom_vmap(*fun*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/custom_batching.py#L47-L174)[\#](#jax.custom_batching.custom_vmap "Link to this definition")  
Customize the vmap behavior of a JAX-transformable function.

This decorator is used to customize the behavior of a JAX function under the [`jax.vmap()`](jax.vmap.html#jax.vmap "jax.vmap") transformation. A `custom_vmap`-decorated function will mostly (see below for caveats) have the same behavior as the underlying function, except when batched using [`jax.vmap()`](jax.vmap.html#jax.vmap "jax.vmap"). When batched, the rule defined using [`def_vmap()`](jax.custom_batching.custom_vmap.def_vmap.html#jax.custom_batching.custom_vmap.def_vmap "jax.custom_batching.custom_vmap.def_vmap") will be used.

For example:

    >>> @jax.custom_batching.custom_vmap
    ... def f(x, y):
    ...   return x + y
    ...
    >>> @f.def_vmap
    ... def f_vmap_rule(axis_size, in_batched, xs, ys):
    ...   assert all(in_batched)
    ...   assert xs.shape[0] == axis_size
    ...   assert ys.shape[0] == axis_size
    ...   out_batched = True
    ...   return xs * ys, out_batched
    ...
    >>> xs = jnp.arange(3)
    >>> ys = jnp.arange(1, 4)
    >>> jax.vmap(f)(xs, ys)  # prints xs * ys instead of xs + ys
    Array([0, 2, 6], dtype=int32)

Of note, `custom_vmap` functions do not support reverse-mode autodiff. To customize both vmap and reverse-mode autodiff, combine `custom_vmap` with [`jax.custom_vjp`](jax.custom_vjp.html#jax.custom_vjp "jax.custom_vjp"). For example:

    >>> @jax.custom_vjp
    ... @jax.custom_batching.custom_vmap
    ... def f(x, y):
    ...   return jnp.sin(x) * y
    ...
    >>> @f.def_vmap
    ... def f_vmap_rule(axis_size, in_batched, xs, ys):
    ...   return jnp.cos(xs) * ys, True
    ...
    >>> def f_fwd(x, y):
    ...   return f(x, y), (jnp.cos(x), jnp.sin(x), y)
    ...
    >>> def f_bwd(res, g):
    ...   cos_x, sin_x, y = res
    ...   return (cos_x * g * y, sin_x * g)
    ...
    >>> f.defvjp(f_fwd, f_bwd)
    >>> jax.vmap(f)(jnp.zeros(3), jnp.ones(3))
    Array([1., 1., 1.], dtype=float32)
    >>> jax.grad(f)(jnp.zeros(()), jnp.ones(()))
    Array(1., dtype=float32)

Note that the [`jax.custom_vjp`](jax.custom_vjp.html#jax.custom_vjp "jax.custom_vjp") must be on the outside, wrapping the `custom_vmap`-decorated function.

Parameters:  
**fun** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[...,* *Any\]*)

\_\_init\_\_(*fun*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/custom_batching.py#L110-L114)[\#](#jax.custom_batching.custom_vmap.__init__ "Link to this definition")  
Parameters:  
**fun** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[...,* *Any\]*)

Methods

|  |  |
|----|----|
| [`__init__`](#jax.custom_batching.custom_vmap.__init__ "jax.custom_batching.custom_vmap.__init__")(fun) |  |
| [`def_vmap`](jax.custom_batching.custom_vmap.def_vmap.html#jax.custom_batching.custom_vmap.def_vmap "jax.custom_batching.custom_vmap.def_vmap")(vmap_rule) | Define the vmap rule for this custom_vmap function. |

Attributes

|             |     |
|-------------|-----|
| `fun`       |     |
| `vmap_rule` |     |

[](jax.custom_vjp.defvjp.html "previous page")

previous

jax.custom_vjp.defvjp

[](jax.custom_batching.custom_vmap.def_vmap.html "next page")

next

jax.custom_batching.custom_vmap.def_vmap

Contents

- [`custom_vmap`](#jax.custom_batching.custom_vmap)
  - [`custom_vmap.__init__()`](#jax.custom_batching.custom_vmap.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
