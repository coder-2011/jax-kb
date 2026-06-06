- [](../index.html)
- [API Reference](../jax.html)
- [`jax.nn` module](../jax.nn.html)
- jax.nn.silu

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.nn.silu.rst "Download source file")
-  .pdf

# jax.nn.silu

## Contents

- [`silu()`](#jax.nn.silu)

# jax.nn.silu[\#](#jax-nn-silu "Link to this heading")

jax.nn.silu(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/nn/functions.py#L234-L256)[\#](#jax.nn.silu "Link to this definition")  
SiLU (aka swish) activation function.

Computes the element-wise function:

\\\mathrm{silu}(x) = x \cdot \mathrm{sigmoid}(x) = \frac{x}{1 + e^{-x}}\\

[`swish()`](jax.nn.swish.html#jax.nn.swish "jax.nn.swish") and [`silu()`](#jax.nn.silu "jax.nn.silu") are both aliases for the same function.

Parameters:  
**x** (*ArrayLike*) – input array

Returns:  
An array.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

[`sigmoid()`](jax.nn.sigmoid.html#jax.nn.sigmoid "jax.nn.sigmoid")

[](jax.nn.soft_sign.html "previous page")

previous

jax.nn.soft_sign

[](jax.nn.swish.html "next page")

next

jax.nn.swish

Contents

- [`silu()`](#jax.nn.silu)

By The JAX authors

© Copyright 2024, The JAX Authors.\
