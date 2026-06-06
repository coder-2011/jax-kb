- [](../index.html)
- [API Reference](../jax.html)
- jax.custom_vjp.defvjp

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.custom_vjp.defvjp.rst "Download source file")
-  .pdf

# jax.custom_vjp.defvjp

## Contents

- [`custom_vjp.defvjp()`](#jax.custom_vjp.defvjp)

# jax.custom_vjp.defvjp[\#](#jax-custom-vjp-defvjp "Link to this heading")

custom_vjp.defvjp(*fwd*, *bwd*, *symbolic_zeros=False*, *optimize_remat=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/custom_derivatives.py#L618-L715)[\#](#jax.custom_vjp.defvjp "Link to this definition")  
Define a custom VJP rule for the function represented by this instance.

Parameters:  
- **fwd** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[...,* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[ReturnValue,* *Any\]\]*) – a Python callable representing the forward pass of the custom VJP rule. When there are no `nondiff_argnums`, the `fwd` function has the same input signature as the underlying primal function. It should return as output a pair, where the first element represents the primal output and the second element represents any “residual” values to store from the forward pass for use on the backward pass by the function `bwd`. Input arguments and elements of the output pair may be arrays or nested tuples/lists/dicts thereof.

- **bwd** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[...,* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[Any,* *...\]\]*) – a Python callable representing the backward pass of the custom VJP rule. When there are no `nondiff_argnums`, the `bwd` function takes two arguments, where the first is the “residual” values produced on the forward pass by `fwd`, and the second is the output cotangent with the same structure as the primal function output. The output of `bwd` must be a tuple of length equal to the number of arguments of the primal function, and the tuple elements may be arrays or nested tuples/lists/dicts thereof so as to match the structure of the primal input arguments.

- **symbolic_zeros** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) –

  boolean, determining whether to indicate symbolic zeros to the `fwd` and `bwd` rules. Enabling this option allows custom derivative rules to detect when certain inputs, and when certain output cotangents, are not involved in differentiation. If `True`:

  - `fwd` must accept, in place of each leaf value `x` in the pytree comprising an argument to the original function, an object (of type `jax.custom_derivatives.CustomVJPPrimal`) with two attributes instead: `value` and `perturbed`. The `value` field is the original primal argument, and `perturbed` is a boolean. The `perturbed` bit indicates whether the argument is involved in differentiation (i.e., if it is `False`, then the corresponding Jacobian “column” is zero).

  - `bwd` will be passed objects representing static symbolic zeros in its cotangent argument in correspondence with unperturbed values; otherwise, only standard JAX types (e.g. array-likes) are passed.

  Setting this option to `True` allows these rules to detect whether certain inputs and outputs are not involved in differentiation, but at the cost of special handling. For instance:

  - The signature of `fwd` changes, and the objects it is passed cannot be output from the rule directly.

  - The `bwd` rule is passed objects that are not entirely array-like, and that cannot be passed to most `jax.numpy` functions.

  - Any custom pytree nodes involved in the primal function’s arguments must accept, in their unflattening functions, the two-field record objects that are given as input leaves to the `fwd` rule.

  Default `False`.

- **optimize_remat** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – boolean, an experimental flag to enable an automatic optimization when this function is used under `jax.remat()`. This will be most useful when the `fwd` rule is an opaque call such as a Pallas kernel or a custom call. Default `False`.

Returns:  
None.

Return type:  
None

Examples

    >>> @jax.custom_vjp
    ... def f(x, y):
    ...   return jnp.sin(x) * y
    ...
    >>> def f_fwd(x, y):
    ...   return f(x, y), (jnp.cos(x), jnp.sin(x), y)
    ...
    >>> def f_bwd(res, g):
    ...   cos_x, sin_x, y = res
    ...   return (cos_x * g * y, sin_x * g)
    ...
    >>> f.defvjp(f_fwd, f_bwd)

    >>> x = jnp.float32(1.0)
    >>> y = jnp.float32(2.0)
    >>> with jnp.printoptions(precision=2):
    ...   print(jax.value_and_grad(f)(x, y))
    (Array(1.68, dtype=float32), Array(1.08, dtype=float32))

[](jax.custom_vjp.html "previous page")

previous

jax.custom_vjp

[](jax.custom_batching.custom_vmap.html "next page")

next

jax.custom_batching.custom_vmap

Contents

- [`custom_vjp.defvjp()`](#jax.custom_vjp.defvjp)

By The JAX authors

© Copyright 2024, The JAX Authors.\
