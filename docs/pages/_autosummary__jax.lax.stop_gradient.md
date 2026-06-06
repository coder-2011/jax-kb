- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.stop_gradient

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.stop_gradient.rst "Download source file")
-  .pdf

# jax.lax.stop_gradient

## Contents

- [`stop_gradient()`](#jax.lax.stop_gradient)

# jax.lax.stop_gradient[\#](#jax-lax-stop-gradient "Link to this heading")

jax.lax.stop_gradient(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L3706-L3752)[\#](#jax.lax.stop_gradient "Link to this definition")  
Stops gradient computation.

Operationally `stop_gradient` is the identity function, that is, it returns argument x unchanged. However, `stop_gradient` prevents the flow of gradients during forward or reverse-mode automatic differentiation. If there are multiple nested gradient computations, `stop_gradient` stops gradients for all of them. For some discussion of where this is useful, refer to [Stopping gradients](../higher-order.html#stopping-gradients).

Parameters:  
**x** (*T*) – array or pytree of arrays

Returns:  
input value is returned unchanged, but within autodiff will be treated as a constant.

Return type:  
T

Examples

Consider a simple function that returns the square of the input value:

    >>> def f1(x):
    ...   return x ** 2
    >>> x = jnp.float32(3.0)
    >>> f1(x)
    Array(9.0, dtype=float32)
    >>> jax.grad(f1)(x)
    Array(6.0, dtype=float32)

The same function with `stop_gradient` around `x` will be equivalent under normal evaluation, but return a zero gradient because `x` is effectively treated as a constant:

    >>> def f2(x):
    ...   return jax.lax.stop_gradient(x) ** 2
    >>> f2(x)
    Array(9.0, dtype=float32)
    >>> jax.grad(f2)(x)
    Array(0.0, dtype=float32)

This is used in a number of places within the JAX codebase; for example [`jax.nn.softmax()`](jax.nn.softmax.html#jax.nn.softmax "jax.nn.softmax") internally normalizes the input by its maximum value, and this maximum value is wrapped in `stop_gradient` for efficiency. Refer to [Stopping gradients](../higher-order.html#stopping-gradients) for more discussion of the applicability of `stop_gradient`.

[](jax.lax.while_loop.html "previous page")

previous

jax.lax.while_loop

[](jax.lax.custom_linear_solve.html "next page")

next

jax.lax.custom_linear_solve

Contents

- [`stop_gradient()`](#jax.lax.stop_gradient)

By The JAX authors

© Copyright 2024, The JAX Authors.\
