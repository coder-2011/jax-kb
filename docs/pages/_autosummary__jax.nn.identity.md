- [](../index.html)
- [API Reference](../jax.html)
- [`jax.nn` module](../jax.nn.html)
- jax.nn.identity

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.nn.identity.rst "Download source file")
-  .pdf

# jax.nn.identity

## Contents

- [`identity()`](#jax.nn.identity)

# jax.nn.identity[\#](#jax-nn-identity "Link to this heading")

jax.nn.identity(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/nn/functions.py#L52-L70)[\#](#jax.nn.identity "Link to this definition")  
Identity activation function.

Returns the argument unmodified.

Parameters:  
**x** (*ArrayLike*) – input array

Returns:  
The argument x unmodified.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Examples

    >>> jax.nn.identity(jax.numpy.array([-2., -1., -0.5, 0, 0.5, 1., 2.]))
    Array([-2. , -1. , -0.5, 0. , 0.5, 1. , 2. ], dtype=float32)

[](jax.nn.mish.html "previous page")

previous

jax.nn.mish

[](jax.nn.softmax.html "next page")

next

jax.nn.softmax

Contents

- [`identity()`](#jax.nn.identity)

By The JAX authors

© Copyright 2024, The JAX Authors.\
