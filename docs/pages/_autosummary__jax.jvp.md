- [](../index.html)
- [API Reference](../jax.html)
- jax.jvp

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.jvp.rst "Download source file")
-  .pdf

# jax.jvp

## Contents

- [`jvp()`](#jax.jvp)

# jax.jvp[\#](#jax-jvp "Link to this heading")

jax.jvp(*fun*, *primals*, *tangents*, *has_aux=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/api.py#L1344-L1391)[\#](#jax.jvp "Link to this definition")  
Computes a (forward-mode) Jacobian-vector product of `fun`.

Parameters:  
- **fun** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")) – Function to be differentiated. Its arguments should be arrays, scalars, or standard Python containers of arrays or scalars. It should return an array, scalar, or standard Python container of arrays or scalars.

- **primals** – The primal values at which the Jacobian of `fun` should be evaluated. Should be either a tuple or a list of arguments, and its length should be equal to the number of positional parameters of `fun`.

- **tangents** – The tangent vector for which the Jacobian-vector product should be evaluated. Should be either a tuple or a list of tangents, with the same tree structure and array shapes as `primals`.

- **has_aux** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – Optional, bool. Indicates whether `fun` returns a pair where the first element is considered the output of the mathematical function to be differentiated and the second element is auxiliary data. Default False.

Returns:  
If `has_aux` is `False`, returns a `(primals_out,`` ``tangents_out)` pair, where `primals_out` is `fun(*primals)`, and `tangents_out` is the Jacobian-vector product of `function` evaluated at `primals` with `tangents`. The `tangents_out` value has the same Python tree structure and shapes as `primals_out`. If `has_aux` is `True`, returns a `(primals_out,`` ``tangents_out,`` ``aux)` tuple where `aux` is the auxiliary data returned by `fun`.

Return type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[Any, …\]

For example:

    >>> import jax
    >>>
    >>> primals, tangents = jax.jvp(jax.numpy.sin, (0.1,), (0.2,))
    >>> print(primals)
    0.09983342
    >>> print(tangents)
    0.19900084

[](jax.hessian.html "previous page")

previous

jax.hessian

[](jax.linearize.html "next page")

next

jax.linearize

Contents

- [`jvp()`](#jax.jvp)

By The JAX authors

© Copyright 2024, The JAX Authors.\
