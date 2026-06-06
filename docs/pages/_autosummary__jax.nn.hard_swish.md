- [](../index.html)
- [API Reference](../jax.html)
- [`jax.nn` module](../jax.nn.html)
- jax.nn.hard_swish

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.nn.hard_swish.rst "Download source file")
-  .pdf

# jax.nn.hard_swish

## Contents

- [`hard_swish()`](#jax.nn.hard_swish)

# jax.nn.hard_swish[\#](#jax-nn-hard-swish "Link to this heading")

jax.nn.hard_swish(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/nn/functions.py#L833-L856)[\#](#jax.nn.hard_swish "Link to this definition")  
Hard SiLU (swish) activation function

Computes the element-wise function

\\\mathrm{hard\\silu}(x) = x \cdot \mathrm{hard\\sigmoid}(x)\\

Both [`hard_silu()`](jax.nn.hard_silu.html#jax.nn.hard_silu "jax.nn.hard_silu") and [`hard_swish()`](#jax.nn.hard_swish "jax.nn.hard_swish") are aliases for the same function.

Parameters:  
**x** (*ArrayLike*) – input array

Returns:  
An array.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

[`hard_sigmoid()`](jax.nn.hard_sigmoid.html#jax.nn.hard_sigmoid "jax.nn.hard_sigmoid")

[](jax.nn.hard_silu.html "previous page")

previous

jax.nn.hard_silu

[](jax.nn.hard_tanh.html "next page")

next

jax.nn.hard_tanh

Contents

- [`hard_swish()`](#jax.nn.hard_swish)

By The JAX authors

© Copyright 2024, The JAX Authors.\
