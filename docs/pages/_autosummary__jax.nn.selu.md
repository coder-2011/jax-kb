- [](../index.html)
- [API Reference](../jax.html)
- [`jax.nn` module](../jax.nn.html)
- jax.nn.selu

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.nn.selu.rst "Download source file")
-  .pdf

# jax.nn.selu

## Contents

- [`selu()`](#jax.nn.selu)

# jax.nn.selu[\#](#jax-nn-selu "Link to this heading")

jax.nn.selu(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/nn/functions.py#L403-L434)[\#](#jax.nn.selu "Link to this definition")  
Scaled exponential linear unit activation.

Computes the element-wise function:

\\\begin{split}\mathrm{selu}(x) = \lambda \begin{cases} x, & x \> 0\\ \alpha e^x - \alpha, & x \le 0 \end{cases}\end{split}\\

where \\\lambda = 1.0507009873554804934193349852946\\ and \\\alpha = 1.6732632423543772848170429916717\\.

For more information, see [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515).

Parameters:  
**x** (*ArrayLike*) – input array

Returns:  
An array.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

[`elu()`](jax.nn.elu.html#jax.nn.elu "jax.nn.elu")

[](jax.nn.celu.html "previous page")

previous

jax.nn.celu

[](jax.nn.gelu.html "next page")

next

jax.nn.gelu

Contents

- [`selu()`](#jax.nn.selu)

By The JAX authors

© Copyright 2024, The JAX Authors.\
