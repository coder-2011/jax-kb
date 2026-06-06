- [](../index.html)
- [API Reference](../jax.html)
- [`jax.nn` module](../jax.nn.html)
- jax.nn.sparse_plus

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.nn.sparse_plus.rst "Download source file")
-  .pdf

# jax.nn.sparse_plus

## Contents

- [`sparse_plus()`](#jax.nn.sparse_plus)

# jax.nn.sparse_plus[\#](#jax-nn-sparse-plus "Link to this heading")

jax.nn.sparse_plus(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/nn/functions.py#L141-L165)[\#](#jax.nn.sparse_plus "Link to this definition")  
Sparse plus function.

Computes the function:

\\\begin{split}\mathrm{sparse\\plus}(x) = \begin{cases} 0, & x \leq -1\\ \frac{1}{4}(x+1)^2, & -1 \< x \< 1 \\ x, & 1 \leq x \end{cases}\end{split}\\

This is the twin function of the softplus activation ensuring a zero output for inputs less than -1 and a linear output for inputs greater than 1, while remaining smooth, convex, monotonic by an adequate definition between -1 and 1.

Parameters:  
**x** (*ArrayLike*) – input (float)

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.nn.softplus.html "previous page")

previous

jax.nn.softplus

[](jax.nn.sparse_sigmoid.html "next page")

next

jax.nn.sparse_sigmoid

Contents

- [`sparse_plus()`](#jax.nn.sparse_plus)

By The JAX authors

© Copyright 2024, The JAX Authors.\
