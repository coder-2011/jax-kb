- [](../index.html)
- [API Reference](../jax.html)
- [`jax.nn` module](../jax.nn.html)
- jax.nn.celu

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.nn.celu.rst "Download source file")
-  .pdf

# jax.nn.celu

## Contents

- [`celu()`](#jax.nn.celu)

# jax.nn.celu[\#](#jax-nn-celu "Link to this heading")

jax.nn.celu(*x*, *alpha=1.0*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/nn/functions.py#L378-L402)[\#](#jax.nn.celu "Link to this definition")  
Continuously-differentiable exponential linear unit activation.

Computes the element-wise function:

\\\begin{split}\mathrm{celu}(x) = \begin{cases} x, & x \> 0\\ \alpha \left(\exp(\frac{x}{\alpha}) - 1\right), & x \le 0 \end{cases}\end{split}\\

For more information, see [Continuously Differentiable Exponential Linear Units](https://arxiv.org/abs/1704.07483).

Parameters:  
- **x** (*ArrayLike*) – input array

- **alpha** (*ArrayLike*) – array or scalar (default: 1.0)

Returns:  
An array.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.nn.elu.html "previous page")

previous

jax.nn.elu

[](jax.nn.selu.html "next page")

next

jax.nn.selu

Contents

- [`celu()`](#jax.nn.celu)

By The JAX authors

© Copyright 2024, The JAX Authors.\
