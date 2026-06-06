- [](../index.html)
- [API Reference](../jax.html)
- jax.custom_vjp

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.custom_vjp.rst "Download source file")
-  .pdf

# jax.custom_vjp

## Contents

- [`custom_vjp`](#jax.custom_vjp)
  - [`custom_vjp.__init__()`](#jax.custom_vjp.__init__)

# jax.custom_vjp[\#](#jax-custom-vjp "Link to this heading")

*class* jax.custom_vjp(*fun*, *nondiff_argnums=()*, *nondiff_argnames=()*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/custom_derivatives.py#L549-L775)[\#](#jax.custom_vjp "Link to this definition")  
Set up a JAX-transformable function for a custom VJP rule definition.

This class is meant to be used as a function decorator. Instances are callables that behave similarly to the underlying function to which the decorator was applied, except when a reverse-mode differentiation transformation (like [`jax.grad()`](jax.grad.html#jax.grad "jax.grad")) is applied, in which case a custom user-supplied VJP rule function is used instead of tracing into and performing automatic differentiation of the underlying function’s implementation. There is a single instance method, [`defvjp()`](jax.custom_vjp.defvjp.html#jax.custom_vjp.defvjp "jax.custom_vjp.defvjp"), which may be used to define the custom VJP rule.

This decorator precludes the use of forward-mode automatic differentiation.

For example:

    @jax.custom_vjp
    def f(x, y):
      return jnp.sin(x) * y

    def f_fwd(x, y):
      return f(x, y), (jnp.cos(x), jnp.sin(x), y)

    def f_bwd(res, g):
      cos_x, sin_x, y = res
      return (cos_x * g * y, sin_x * g)

    f.defvjp(f_fwd, f_bwd)

For a more detailed introduction, see the [tutorial](https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html).

\_\_init\_\_(*fun*, *nondiff_argnums=()*, *nondiff_argnames=()*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/custom_derivatives.py#L591-L615)[\#](#jax.custom_vjp.__init__ "Link to this definition")  
Parameters:  
- **fun** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[...,* *ReturnValue\]*)

- **nondiff_argnums** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*)

- **nondiff_argnames** (*Sequence\[*[*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*\]*)

Methods

|  |  |
|----|----|
| [`__init__`](#jax.custom_vjp.__init__ "jax.custom_vjp.__init__")(fun\[, nondiff_argnums, ...\]) |  |
| [`defvjp`](jax.custom_vjp.defvjp.html#jax.custom_vjp.defvjp "jax.custom_vjp.defvjp")(fwd, bwd\[, symbolic_zeros, ...\]) | Define a custom VJP rule for the function represented by this instance. |

[](jax.custom_jvp.defjvps.html "previous page")

previous

jax.custom_jvp.defjvps

[](jax.custom_vjp.defvjp.html "next page")

next

jax.custom_vjp.defvjp

Contents

- [`custom_vjp`](#jax.custom_vjp)
  - [`custom_vjp.__init__()`](#jax.custom_vjp.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
