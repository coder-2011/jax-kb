- [](../index.html)
- [API Reference](../jax.html)
- [`jax.nn` module](../jax.nn.html)
- jax.nn.sparse_sigmoid

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.nn.sparse_sigmoid.rst "Download source file")
-  .pdf

# jax.nn.sparse_sigmoid

## Contents

- [`sparse_sigmoid()`](#jax.nn.sparse_sigmoid)

# jax.nn.sparse_sigmoid[\#](#jax-nn-sparse-sigmoid "Link to this heading")

jax.nn.sparse_sigmoid(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/nn/functions.py#L202-L233)[\#](#jax.nn.sparse_sigmoid "Link to this definition")  
Sparse sigmoid activation function.

Computes the function:

\\\begin{split}\mathrm{sparse\\sigmoid}(x) = \begin{cases} 0, & x \leq -1\\ \frac{1}{2}(x+1), & -1 \< x \< 1 \\ 1, & 1 \leq x \end{cases}\end{split}\\

This is the twin function of the `sigmoid` activation ensuring a zero output for inputs less than -1, a 1 output for inputs greater than 1, and a linear output for inputs between -1 and 1. It is the derivative of `sparse_plus`.

For more information, see [Learning with Fenchel-Young Losses (section 6.2)](https://arxiv.org/abs/1901.02324).

Parameters:  
**x** (*ArrayLike*) – input array

Returns:  
An array.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

[`sigmoid()`](jax.nn.sigmoid.html#jax.nn.sigmoid "jax.nn.sigmoid")

[](jax.nn.sparse_plus.html "previous page")

previous

jax.nn.sparse_plus

[](jax.nn.soft_sign.html "next page")

next

jax.nn.soft_sign

Contents

- [`sparse_sigmoid()`](#jax.nn.sparse_sigmoid)

By The JAX authors

© Copyright 2024, The JAX Authors.\
