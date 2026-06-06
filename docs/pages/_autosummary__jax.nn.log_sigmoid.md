- [](../index.html)
- [API Reference](../jax.html)
- [`jax.nn` module](../jax.nn.html)
- jax.nn.log_sigmoid

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.nn.log_sigmoid.rst "Download source file")
-  .pdf

# jax.nn.log_sigmoid

## Contents

- [`log_sigmoid()`](#jax.nn.log_sigmoid)

# jax.nn.log_sigmoid[\#](#jax-nn-log-sigmoid "Link to this heading")

jax.nn.log_sigmoid(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/nn/functions.py#L281-L301)[\#](#jax.nn.log_sigmoid "Link to this definition")  
Log-sigmoid activation function.

Computes the element-wise function:

\\\mathrm{log\\sigmoid}(x) = \log(\mathrm{sigmoid}(x)) = -\log(1 + e^{-x})\\

Parameters:  
**x** (*ArrayLike*) – input array

Returns:  
An array.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

[`sigmoid()`](jax.nn.sigmoid.html#jax.nn.sigmoid "jax.nn.sigmoid")

[](jax.nn.swish.html "previous page")

previous

jax.nn.swish

[](jax.nn.leaky_relu.html "next page")

next

jax.nn.leaky_relu

Contents

- [`log_sigmoid()`](#jax.nn.log_sigmoid)

By The JAX authors

© Copyright 2024, The JAX Authors.\
