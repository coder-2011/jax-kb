- [](../index.html)
- [API Reference](../jax.html)
- jax.custom_jvp.defjvp

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.custom_jvp.defjvp.rst "Download source file")
-  .pdf

# jax.custom_jvp.defjvp

## Contents

- [`custom_jvp.defjvp()`](#jax.custom_jvp.defjvp)

# jax.custom_jvp.defjvp[\#](#jax-custom-jvp-defjvp "Link to this heading")

custom_jvp.defjvp(*jvp*, *symbolic_zeros=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/custom_derivatives.py#L186-L237)[\#](#jax.custom_jvp.defjvp "Link to this definition")  
Define a custom JVP rule for the function represented by this instance.

Parameters:  
- **jvp** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[...,* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[ReturnValue,* *ReturnValue\]\]*) – a Python callable representing the custom JVP rule. When there are no `nondiff_argnums`, the `jvp` function should accept two arguments, where the first is a tuple of primal inputs and the second is a tuple of tangent inputs. The lengths of both tuples are equal to the number of parameters of the [`custom_jvp`](jax.custom_jvp.html#jax.custom_jvp "jax.custom_jvp") function. The `jvp` function should produce as output a pair where the first element is the primal output and the second element is the tangent output. Elements of the input and output tuples may be arrays or any nested tuples/lists/dicts thereof.

- **symbolic_zeros** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – boolean, indicating whether the rule should be passed objects representing static symbolic zeros in its tangent argument in correspondence with unperturbed values; otherwise, only standard JAX types (e.g. array-likes) are passed. Setting this option to `True` allows a JVP rule to detect whether certain inputs are not involved in differentiation, but at the cost of needing special handling for these objects (which e.g. can’t be passed into jax.numpy functions). Default `False`.

Returns:  
Returns `jvp` so that `defjvp` can be used as a decorator.

Return type:  
[Callable](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")\[…, [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[ReturnValue, ReturnValue\]\]

Examples

    >>> @jax.custom_jvp
    ... def f(x, y):
    ...   return jnp.sin(x) * y
    ...
    >>> @f.defjvp
    ... def f_jvp(primals, tangents):
    ...   x, y = primals
    ...   x_dot, y_dot = tangents
    ...   primal_out = f(x, y)
    ...   tangent_out = jnp.cos(x) * x_dot * y + jnp.sin(x) * y_dot
    ...   return primal_out, tangent_out

    >>> x = jnp.float32(1.0)
    >>> y = jnp.float32(2.0)
    >>> with jnp.printoptions(precision=2):
    ...   print(jax.value_and_grad(f)(x, y))
    (Array(1.68, dtype=float32), Array(1.08, dtype=float32))

[](jax.custom_jvp.html "previous page")

previous

jax.custom_jvp

[](jax.custom_jvp.defjvps.html "next page")

next

jax.custom_jvp.defjvps

Contents

- [`custom_jvp.defjvp()`](#jax.custom_jvp.defjvp)

By The JAX authors

© Copyright 2024, The JAX Authors.\
