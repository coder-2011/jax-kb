- [](../index.html)
- [API Reference](../jax.html)
- jax.make_jaxpr

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.make_jaxpr.rst "Download source file")
-  .pdf

# jax.make_jaxpr

## Contents

- [`make_jaxpr()`](#jax.make_jaxpr)

# jax.make_jaxpr[\#](#jax-make-jaxpr "Link to this heading")

jax.make_jaxpr(*fun: [Callable](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*, *static_argnums: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") \| Sequence\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")\] = ()*, *axis_env: Sequence\[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[AxisName, [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")\]\] \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *return_shape: [Literal](jax.extend.core.Literal.html#jax.extend.core.Literal "jax.extend.core.Literal")\[False\] = False*) → [Callable](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")\[..., [core.ClosedJaxpr](jax.extend.core.ClosedJaxpr.html#jax.extend.core.ClosedJaxpr "jax.extend.core.ClosedJaxpr")\][\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/api.py#L1877-L1969)[\#](#jax.make_jaxpr "Link to this definition")\
jax.make_jaxpr(*fun: [Callable](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*, *static_argnums: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") \| Sequence\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")\] = ()*, *axis_env: Sequence\[[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[AxisName, [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")\]\] \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *return_shape: [Literal](jax.extend.core.Literal.html#jax.extend.core.Literal "jax.extend.core.Literal")\[True\] = False*) → [Callable](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")\[..., [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[core.ClosedJaxpr](jax.extend.core.ClosedJaxpr.html#jax.extend.core.ClosedJaxpr "jax.extend.core.ClosedJaxpr"), Any\]\]  
Create a function that returns the jaxpr of `fun` given example args.

Parameters:  
- **fun** – The function whose `jaxpr` is to be computed. Its positional arguments and return value should be arrays, scalars, or standard Python containers (tuple/list/dict) thereof.

- **static_argnums** – See the [`jax.jit()`](jax.jit.html#jax.jit "jax.jit") docstring.

- **axis_env** – Optional, a sequence of pairs where the first element is an axis name and the second element is a positive integer representing the size of the mapped axis with that name. This parameter is useful when lowering functions that involve parallel communication collectives, and it specifies the axis name/size environment that would be set up by applications of [`jax.pmap()`](jax.pmap.html#jax.pmap "jax.pmap").

- **return_shape** – Optional boolean, defaults to `False`. If `True`, the wrapped function returns a pair where the first element is the `ClosedJaxpr` representation of `fun` and the second element is a pytree with the same structure as the output of `fun` and where the leaves are objects with `shape` and `dtype` attributes representing the corresponding types of the output leaves.

Returns:  
A wrapped version of `fun` that when applied to example arguments returns a `ClosedJaxpr` representation of `fun` on those arguments. If the argument `return_shape` is `True`, then the returned function instead returns a pair where the first element is the `ClosedJaxpr` representation of `fun` and the second element is a pytree representing the structure, shape, dtypes, and named shapes of the output of `fun`.

A `jaxpr` is JAX’s intermediate representation for program traces. The `jaxpr` language is based on the simply-typed first-order lambda calculus with let-bindings. [`make_jaxpr()`](#jax.make_jaxpr "jax.make_jaxpr") adapts a function to return its `jaxpr`, which we can inspect to understand what JAX is doing internally. The `jaxpr` returned is a trace of `fun` abstracted to `ShapedArray` level. Other levels of abstraction exist internally.

We do not describe the semantics of the `jaxpr` language in detail here, but instead give a few examples.

    >>> import jax
    >>>
    >>> def f(x): return jax.numpy.sin(jax.numpy.cos(x))
    >>> print(f(3.0))
    -0.83602
    >>> jax.make_jaxpr(f)(3.0)
    { lambda ; a:f32[]. let b:f32[] = cos a; c:f32[] = sin b in (c,) }
    >>> jax.make_jaxpr(jax.grad(f))(3.0)
    { lambda ; a:f32[]. let
        b:f32[] = cos a
        c:f32[] = sin a
        _:f32[] = sin b
        d:f32[] = cos b
        e:f32[] = mul 1.0:f32[] d
        f:f32[] = neg e
        g:f32[] = mul f c
      in (g,) }

[](jax.ensure_compile_time_eval.html "previous page")

previous

jax.ensure_compile_time_eval

[](jax.eval_shape.html "next page")

next

jax.eval_shape

Contents

- [`make_jaxpr()`](#jax.make_jaxpr)

By The JAX authors

© Copyright 2024, The JAX Authors.\
