- [](../index.html)
- [API Reference](../jax.html)
- [`jax.nn` module](../jax.nn.html)
- jax.nn.leaky_relu

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.nn.leaky_relu.rst "Download source file")
-  .pdf

# jax.nn.leaky_relu

## Contents

- [`leaky_relu()`](#jax.nn.leaky_relu)

# jax.nn.leaky_relu[\#](#jax-nn-leaky-relu "Link to this heading")

jax.nn.leaky_relu(*x*, *negative_slope=0.01*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/nn/functions.py#L329-L355)[\#](#jax.nn.leaky_relu "Link to this definition")  
Leaky rectified linear unit activation function.

Computes the element-wise function:

\\\begin{split}\mathrm{leaky\\relu}(x) = \begin{cases} x, & x \ge 0\\ \alpha x, & x \< 0 \end{cases}\end{split}\\

where \\\alpha\\ = `negative_slope`.

Parameters:  
- **x** (*ArrayLike*) – input array

- **negative_slope** (*ArrayLike*) – array or scalar specifying the negative slope (default: 0.01)

Returns:  
An array.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

[`relu()`](jax.nn.relu.html#jax.nn.relu "jax.nn.relu")

[](jax.nn.log_sigmoid.html "previous page")

previous

jax.nn.log_sigmoid

[](jax.nn.hard_sigmoid.html "next page")

next

jax.nn.hard_sigmoid

Contents

- [`leaky_relu()`](#jax.nn.leaky_relu)

By The JAX authors

© Copyright 2024, The JAX Authors.\
