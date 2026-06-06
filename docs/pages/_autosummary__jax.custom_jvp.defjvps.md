- [](../index.html)
- [API Reference](../jax.html)
- jax.custom_jvp.defjvps

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.custom_jvp.defjvps.rst "Download source file")
-  .pdf

# jax.custom_jvp.defjvps

## Contents

- [`custom_jvp.defjvps()`](#jax.custom_jvp.defjvps)

# jax.custom_jvp.defjvps[\#](#jax-custom-jvp-defjvps "Link to this heading")

custom_jvp.defjvps(*\*jvps*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/custom_derivatives.py#L238-L279)[\#](#jax.custom_jvp.defjvps "Link to this definition")  
Convenience wrapper for defining JVPs for each argument separately.

This convenience wrapper cannot be used together with `nondiff_argnums`.

Parameters:  
**\*jvps** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[...,* *ReturnValue\]* *\|* *None*) – a sequence of functions, one for each positional argument of the [`custom_jvp`](jax.custom_jvp.html#jax.custom_jvp "jax.custom_jvp") function. Each function takes as arguments the tangent value for the corresponding primal input, the primal output, and the primal inputs. See the example below.

Returns:  
None.

Return type:  
None

Examples

    >>> @jax.custom_jvp
    ... def f(x, y):
    ...   return jnp.sin(x) * y
    ...
    >>> f.defjvps(lambda x_dot, primal_out, x, y: jnp.cos(x) * x_dot * y,
    ...           lambda y_dot, primal_out, x, y: jnp.sin(x) * y_dot)

    >>> x = jnp.float32(1.0)
    >>> y = jnp.float32(2.0)
    >>> with jnp.printoptions(precision=2):
    ...   print(jax.value_and_grad(f)(x, y))
    (Array(1.68, dtype=float32), Array(1.08, dtype=float32))

[](jax.custom_jvp.defjvp.html "previous page")

previous

jax.custom_jvp.defjvp

[](jax.custom_vjp.html "next page")

next

jax.custom_vjp

Contents

- [`custom_jvp.defjvps()`](#jax.custom_jvp.defjvps)

By The JAX authors

© Copyright 2024, The JAX Authors.\
