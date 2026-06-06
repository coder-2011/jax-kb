- [](../index.html)
- [API Reference](../jax.html)
- [`jax.nn` module](../jax.nn.html)
- jax.nn.mish

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.nn.mish.rst "Download source file")
-  .pdf

# jax.nn.mish

## Contents

- [`mish()`](#jax.nn.mish)

# jax.nn.mish[\#](#jax-nn-mish "Link to this heading")

jax.nn.mish(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/nn/functions.py#L259-L280)[\#](#jax.nn.mish "Link to this definition")  
Mish activation function.

Computes the element-wise function:

\\\mathrm{mish}(x) = x \cdot \mathrm{tanh}(\mathrm{softplus}(x))\\

For more information, see [Mish: A Self Regularized Non-Monotonic Activation Function](https://arxiv.org/abs/1908.08681).

Parameters:  
**x** (*ArrayLike*) – input array

Returns:  
An array.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.nn.squareplus.html "previous page")

previous

jax.nn.squareplus

[](jax.nn.identity.html "next page")

next

jax.nn.identity

Contents

- [`mish()`](#jax.nn.mish)

By The JAX authors

© Copyright 2024, The JAX Authors.\
