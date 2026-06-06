- [](../index.html)
- [API Reference](../jax.html)
- jax.custom_jvp

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.custom_jvp.rst "Download source file")
-  .pdf

# jax.custom_jvp

## Contents

- [`custom_jvp`](#jax.custom_jvp)
  - [`custom_jvp.__init__()`](#jax.custom_jvp.__init__)

# jax.custom_jvp[\#](#jax-custom-jvp "Link to this heading")

*class* jax.custom_jvp(*fun*, *nondiff_argnums=()*, *nondiff_argnames=()*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/custom_derivatives.py#L118-L328)[\#](#jax.custom_jvp "Link to this definition")  
Set up a JAX-transformable function for a custom JVP rule definition.

This class is meant to be used as a function decorator. Instances are callables that behave similarly to the underlying function to which the decorator was applied, except when a differentiation transformation (like [`jax.jvp()`](jax.jvp.html#jax.jvp "jax.jvp") or [`jax.grad()`](jax.grad.html#jax.grad "jax.grad")) is applied, in which case a custom user-supplied JVP rule function is used instead of tracing into and performing automatic differentiation of the underlying function’s implementation.

There are two instance methods available for defining the custom JVP rule: [`defjvp()`](jax.custom_jvp.defjvp.html#jax.custom_jvp.defjvp "jax.custom_jvp.defjvp") for defining a *single* custom JVP rule for all the function’s inputs, and for convenience [`defjvps()`](jax.custom_jvp.defjvps.html#jax.custom_jvp.defjvps "jax.custom_jvp.defjvps"), which wraps [`defjvp()`](jax.custom_jvp.defjvp.html#jax.custom_jvp.defjvp "jax.custom_jvp.defjvp"), and allows you to provide separate definitions for the partial derivatives of the function w.r.t. each of its arguments.

For example:

    @jax.custom_jvp
    def f(x, y):
      return jnp.sin(x) * y

    @f.defjvp
    def f_jvp(primals, tangents):
      x, y = primals
      x_dot, y_dot = tangents
      primal_out = f(x, y)
      tangent_out = jnp.cos(x) * x_dot * y + jnp.sin(x) * y_dot
      return primal_out, tangent_out

For a more detailed introduction, see the [tutorial](https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html).

Parameters:  
- **fun** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[...,* *ReturnValue\]*)

- **nondiff_argnums** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*)

- **nondiff_argnames** (*Sequence\[*[*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*\]*)

\_\_init\_\_(*fun*, *nondiff_argnums=()*, *nondiff_argnames=()*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/custom_derivatives.py#L162-L183)[\#](#jax.custom_jvp.__init__ "Link to this definition")  
Parameters:  
- **fun** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[...,* *ReturnValue\]*)

- **nondiff_argnums** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*)

- **nondiff_argnames** (*Sequence\[*[*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*\]*)

Methods

|  |  |
|----|----|
| [`__init__`](#jax.custom_jvp.__init__ "jax.custom_jvp.__init__")(fun\[, nondiff_argnums, ...\]) |  |
| [`defjvp`](jax.custom_jvp.defjvp.html#jax.custom_jvp.defjvp "jax.custom_jvp.defjvp")(jvp\[, symbolic_zeros\]) | Define a custom JVP rule for the function represented by this instance. |
| [`defjvps`](jax.custom_jvp.defjvps.html#jax.custom_jvp.defjvps "jax.custom_jvp.defjvps")(\*jvps) | Convenience wrapper for defining JVPs for each argument separately. |

Attributes

|                    |     |
|--------------------|-----|
| `jvp`              |     |
| `symbolic_zeros`   |     |
| `fun`              |     |
| `nondiff_argnums`  |     |
| `nondiff_argnames` |     |

[](jax.process_indices.html "previous page")

previous

jax.process_indices

[](jax.custom_jvp.defjvp.html "next page")

next

jax.custom_jvp.defjvp

Contents

- [`custom_jvp`](#jax.custom_jvp)
  - [`custom_jvp.__init__()`](#jax.custom_jvp.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
