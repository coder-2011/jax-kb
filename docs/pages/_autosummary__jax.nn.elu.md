- [](../index.html)
- [API Reference](../jax.html)
- [`jax.nn` module](../jax.nn.html)
- jax.nn.elu

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.nn.elu.rst "Download source file")
-  .pdf

# jax.nn.elu

## Contents

- [`elu()`](#jax.nn.elu)

# jax.nn.elu[\#](#jax-nn-elu "Link to this heading")

jax.nn.elu(*x*, *alpha=1.0*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/nn/functions.py#L302-L328)[\#](#jax.nn.elu "Link to this definition")  
Exponential linear unit activation function.

Computes the element-wise function:

\\\begin{split}\mathrm{elu}(x) = \begin{cases} x, & x \> 0\\ \alpha \left(\exp(x) - 1\right), & x \le 0 \end{cases}\end{split}\\

Parameters:  
- **x** (*ArrayLike*) – input array

- **alpha** (*ArrayLike*) – scalar or array of alpha values (default: 1.0)

Returns:  
An array.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

[`selu()`](jax.nn.selu.html#jax.nn.selu "jax.nn.selu")

[](jax.nn.tanh.html "previous page")

previous

jax.nn.tanh

[](jax.nn.celu.html "next page")

next

jax.nn.celu

Contents

- [`elu()`](#jax.nn.elu)

By The JAX authors

© Copyright 2024, The JAX Authors.\
