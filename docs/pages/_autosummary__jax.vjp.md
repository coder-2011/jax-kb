- [](../index.html)
- [API Reference](../jax.html)
- jax.vjp

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.vjp.rst "Download source file")
-  .pdf

# jax.vjp

## Contents

- [`vjp()`](#jax.vjp)

# jax.vjp[\#](#jax-vjp "Link to this heading")

jax.vjp(*fun: [Callable](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")\[..., T\]*, *\*primals: Any*, *has_aux: [Literal](jax.extend.core.Literal.html#jax.extend.core.Literal "jax.extend.core.Literal")\[False\] = False*, *reduce_axes: Sequence\[AxisName\] = ()*) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[T, [Callable](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")\][\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/api.py#L1561-L1625)[\#](#jax.vjp "Link to this definition")\
jax.vjp(*fun: [Callable](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")\[..., [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[T, U\]\]*, *\*primals: Any*, *has_aux: [Literal](jax.extend.core.Literal.html#jax.extend.core.Literal "jax.extend.core.Literal")\[True\]*, *reduce_axes: Sequence\[AxisName\] = ()*) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[T, [Callable](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable"), U\]  
Compute a (reverse-mode) vector-Jacobian product of `fun`.

[`grad()`](jax.grad.html#jax.grad "jax.grad") is implemented as a special case of [`vjp()`](#jax.vjp "jax.vjp").

Parameters:  
- **fun** – Function to be differentiated. Its arguments should be arrays, scalars, or standard Python containers of arrays or scalars. It should return an array, scalar, or standard Python container of arrays or scalars.

- **primals** – A sequence of primal values at which the Jacobian of `fun` should be evaluated. The number of `primals` should be equal to the number of positional parameters of `fun`. Each primal value should be an array, a scalar, or a pytree (standard Python containers) thereof.

- **has_aux** – Optional, bool. Indicates whether `fun` returns a pair where the first element is considered the output of the mathematical function to be differentiated and the second element is auxiliary data. Default False.

Returns:  
If `has_aux` is `False`, returns a `(primals_out,`` ``vjpfun)` pair, where `primals_out` is `fun(*primals)`. If `has_aux` is `True`, returns a `(primals_out,`` ``vjpfun,`` ``aux)` tuple where `aux` is the auxiliary data returned by `fun`.

`vjpfun` is a function from a cotangent vector with the same shape as `primals_out` to a tuple of cotangent vectors with the same number and shapes as `primals`, representing the vector-Jacobian product of `fun` evaluated at `primals`.

    >>> import jax
    >>>
    >>> def f(x, y):
    ...   return jax.numpy.sin(x), jax.numpy.cos(y)
    ...
    >>> primals, f_vjp = jax.vjp(f, 0.5, 1.0)
    >>> xbar, ybar = f_vjp((-0.7, 0.3))
    >>> print(xbar)
    -0.61430776
    >>> print(ybar)
    -0.2524413

[](jax.linear_transpose.html "previous page")

previous

jax.linear_transpose

[](jax.custom_gradient.html "next page")

next

jax.custom_gradient

Contents

- [`vjp()`](#jax.vjp)

By The JAX authors

© Copyright 2024, The JAX Authors.\
