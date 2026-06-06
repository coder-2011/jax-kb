- [](../index.html)
- [API Reference](../jax.html)
- [`jax.nn` module](../jax.nn.html)
- jax.nn.squareplus

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.nn.squareplus.rst "Download source file")
-  .pdf

# jax.nn.squareplus

## Contents

- [`squareplus()`](#jax.nn.squareplus)

# jax.nn.squareplus[\#](#jax-nn-squareplus "Link to this heading")

jax.nn.squareplus(*x*, *b=4*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/nn/functions.py#L108-L126)[\#](#jax.nn.squareplus "Link to this definition")  
Squareplus activation function.

Computes the element-wise function

\\\mathrm{squareplus}(x) = \frac{x + \sqrt{x^2 + b}}{2}\\

as described in [https://arxiv.org/abs/2112.11687](https://arxiv.org/abs/2112.11687).

Parameters:  
- **x** (*ArrayLike*) – input array

- **b** (*ArrayLike*) – smoothness parameter

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.nn.glu.html "previous page")

previous

jax.nn.glu

[](jax.nn.mish.html "next page")

next

jax.nn.mish

Contents

- [`squareplus()`](#jax.nn.squareplus)

By The JAX authors

© Copyright 2024, The JAX Authors.\
