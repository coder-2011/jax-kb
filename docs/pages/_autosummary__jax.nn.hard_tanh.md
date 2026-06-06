- [](../index.html)
- [API Reference](../jax.html)
- [`jax.nn` module](../jax.nn.html)
- jax.nn.hard_tanh

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.nn.hard_tanh.rst "Download source file")
-  .pdf

# jax.nn.hard_tanh

## Contents

- [`hard_tanh()`](#jax.nn.hard_tanh)

# jax.nn.hard_tanh[\#](#jax-nn-hard-tanh "Link to this heading")

jax.nn.hard_tanh(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/nn/functions.py#L356-L377)[\#](#jax.nn.hard_tanh "Link to this definition")  
Hard \\\mathrm{tanh}\\ activation function.

Computes the element-wise function:

\\\begin{split}\mathrm{hard\\tanh}(x) = \begin{cases} -1, & x \< -1\\ x, & -1 \le x \le 1\\ 1, & 1 \< x \end{cases}\end{split}\\

Parameters:  
**x** (*ArrayLike*) – input array

Returns:  
An array.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.nn.hard_swish.html "previous page")

previous

jax.nn.hard_swish

[](jax.nn.tanh.html "next page")

next

jax.nn.tanh

Contents

- [`hard_tanh()`](#jax.nn.hard_tanh)

By The JAX authors

© Copyright 2024, The JAX Authors.\
