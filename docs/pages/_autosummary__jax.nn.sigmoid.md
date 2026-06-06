- [](../index.html)
- [API Reference](../jax.html)
- [`jax.nn` module](../jax.nn.html)
- jax.nn.sigmoid

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.nn.sigmoid.rst "Download source file")
-  .pdf

# jax.nn.sigmoid

## Contents

- [`sigmoid()`](#jax.nn.sigmoid)

# jax.nn.sigmoid[\#](#jax-nn-sigmoid "Link to this heading")

jax.nn.sigmoid(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/nn/functions.py#L181-L201)[\#](#jax.nn.sigmoid "Link to this definition")  
Sigmoid activation function.

Computes the element-wise function:

\\\mathrm{sigmoid}(x) = \frac{1}{1 + e^{-x}}\\

Parameters:  
**x** (*ArrayLike*) – input array

Returns:  
An array.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

[`log_sigmoid()`](jax.nn.log_sigmoid.html#jax.nn.log_sigmoid "jax.nn.log_sigmoid")

[](jax.nn.relu6.html "previous page")

previous

jax.nn.relu6

[](jax.nn.softplus.html "next page")

next

jax.nn.softplus

Contents

- [`sigmoid()`](#jax.nn.sigmoid)

By The JAX authors

© Copyright 2024, The JAX Authors.\
