- [](../index.html)
- [API Reference](../jax.html)
- jax.custom_gradient

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.custom_gradient.rst "Download source file")
-  .pdf

# jax.custom_gradient

## Contents

- [`custom_gradient()`](#jax.custom_gradient)

# jax.custom_gradient[\#](#jax-custom-gradient "Link to this heading")

jax.custom_gradient(*fun*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/custom_derivatives.py#L1157-L1245)[\#](#jax.custom_gradient "Link to this definition")  
Convenience function for defining custom VJP rules (aka custom gradients).

While the canonical way to define custom VJP rules is via `jax.custom_vjp`, the `custom_gradient` convenience wrapper follows TensorFlow’s `tf.custom_gradient` API. The difference here is that `custom_gradient` can be used as a decorator on one function that returns both the primal value (representing the output of the mathematical function to be differentiated) and the VJP (gradient) function. See [https://www.tensorflow.org/api_docs/python/tf/custom_gradient](https://www.tensorflow.org/api_docs/python/tf/custom_gradient).

If the mathematical function to be differentiated has Haskell-like signature `a`` ``->`` ``b`, then the Python callable `fun` should have the signature `a`` ``->`` ``(b,`` ``CT`` ``b`` ``--o`` ``CT`` ``a)` where we use `CT`` ``x` to denote a cotangent type for `x` and the `--o` arrow to denote a linear function. See the example below. That is, `fun` should return a pair where the first element represents the value of the mathematical function to be differentiated and the second element is a function to be called on the backward pass of reverse-mode automatic differentiation (i.e. the “custom gradient” function).

The function returned as the second element of the output of `fun` can close over intermediate values computed when evaluating the function to be differentiated. That is, use lexical closure to share work between the forward pass and the backward pass of reverse-mode automatic differentiation. However, it cannot perform Python control flow which depends on the values of the closed-over intermediate values or its cotangent arguments; if the function includes such control flow, an error is raised.

Parameters:  
**fun** – a Python callable specifying both the mathematical function to be differentiated and its reverse-mode differentiation rule. It should return a pair consisting of an output value and a Python callable that represents the custom gradient function.

Returns:  
A Python callable that accepts the same arguments as `fun` and returns the output value specified by the first element of `fun`’s output pair.

For example:

    >>> @jax.custom_gradient
    ... def f(x):
    ...   return x ** 2, lambda g: (g * x,)
    ...
    >>> print(f(3.))
    9.0
    >>> print(jax.grad(f)(3.))
    3.0

An example with a function on two arguments, so that the VJP function must return a tuple of length two:

    >>> @jax.custom_gradient
    ... def f(x, y):
    ...   return x * y, lambda g: (g * y, g * x)
    ...
    >>> print(f(3., 4.))
    12.0
    >>> print(jax.grad(f, argnums=(0, 1))(3., 4.))
    (Array(4., dtype=float32, weak_type=True), Array(3., dtype=float32, weak_type=True))

[](jax.vjp.html "previous page")

previous

jax.vjp

[](jax.closure_convert.html "next page")

next

jax.closure_convert

Contents

- [`custom_gradient()`](#jax.custom_gradient)

By The JAX authors

© Copyright 2024, The JAX Authors.\
