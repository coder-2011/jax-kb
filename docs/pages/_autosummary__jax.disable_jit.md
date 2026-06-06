- [](../index.html)
- [API Reference](../jax.html)
- jax.disable_jit

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.disable_jit.rst "Download source file")
-  .pdf

# jax.disable_jit

## Contents

- [`disable_jit()`](#jax.disable_jit)

# jax.disable_jit[\#](#jax-disable-jit "Link to this heading")

jax.disable_jit(*disable=True*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/api.py#L362-L411)[\#](#jax.disable_jit "Link to this definition")  
Context manager that disables [`jit()`](jax.jit.html#jax.jit "jax.jit") behavior under its dynamic context.

For debugging, it is useful to have a mechanism that disables [`jit()`](jax.jit.html#jax.jit "jax.jit") everywhere in a dynamic context. Note that this not only disables explicit uses of [`jit()`](jax.jit.html#jax.jit "jax.jit") by the user, but will also remove any implicit JIT compilation used by the JAX library: this includes implicit JIT computation of body and cond functions passed to higher-level primitives like [`scan()`](jax.lax.scan.html#jax.lax.scan "jax.lax.scan") and [`while_loop()`](jax.lax.while_loop.html#jax.lax.while_loop "jax.lax.while_loop"), JIT used in implementations of [`jax.numpy`](../jax.numpy.html#module-jax.numpy "jax.numpy") functions, and any other case where [`jit()`](jax.jit.html#jax.jit "jax.jit") is used within an API’s implementation. Note however that even under disable_jit, individual primitive operations will still be compiled by XLA as in normal eager op-by-op execution.

Values that have a data dependence on the arguments to a jitted function are traced and abstracted. For example, an abstract value may be a `ShapedArray` instance, representing the set of all possible arrays with a given shape and dtype, but not representing one concrete array with specific values. You might notice those if you use a benign side-effecting operation in a jitted function, like a print:

    >>> import jax
    >>>
    >>> @jax.jit
    ... def f(x):
    ...   y = x * 2
    ...   print("Value of y is", y)
    ...   return y + 3
    ...
    >>> print(f(jax.numpy.array([1, 2, 3])))
    Value of y is JitTracer(int32[3])
    [5 7 9]

Here `y` has been abstracted by [`jit()`](jax.jit.html#jax.jit "jax.jit") to a `ShapedArray`, which represents an array with a fixed shape and type but an arbitrary value. The value of `y` is also traced. If we want to see a concrete value while debugging, and avoid the tracer too, we can use the [`disable_jit()`](#jax.disable_jit "jax.disable_jit") context manager:

    >>> import jax
    >>>
    >>> with jax.disable_jit():
    ...   print(f(jax.numpy.array([1, 2, 3])))
    ...
    Value of y is [2 4 6]
    [5 7 9]

Parameters:  
**disable** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

[](jax.jit.html "previous page")

previous

jax.jit

[](jax.ensure_compile_time_eval.html "next page")

next

jax.ensure_compile_time_eval

Contents

- [`disable_jit()`](#jax.disable_jit)

By The JAX authors

© Copyright 2024, The JAX Authors.\
