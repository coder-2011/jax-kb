- [](../index.html)
- [API Reference](../jax.html)
- [`jax.nn` module](../jax.nn.html)
- jax.nn.hard_sigmoid

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.nn.hard_sigmoid.rst "Download source file")
-  .pdf

# jax.nn.hard_sigmoid

## Contents

- [`hard_sigmoid()`](#jax.nn.hard_sigmoid)

# jax.nn.hard_sigmoid[\#](#jax-nn-hard-sigmoid "Link to this heading")

jax.nn.hard_sigmoid(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/nn/functions.py#L813-L832)[\#](#jax.nn.hard_sigmoid "Link to this definition")  
Hard Sigmoid activation function.

Computes the element-wise function

\\\mathrm{hard\\sigmoid}(x) = \frac{\mathrm{relu6}(x + 3)}{6}\\

Parameters:  
**x** (*ArrayLike*) – input array

Returns:  
An array.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

[`relu6()`](jax.nn.relu6.html#jax.nn.relu6 "jax.nn.relu6")

[](jax.nn.leaky_relu.html "previous page")

previous

jax.nn.leaky_relu

[](jax.nn.hard_silu.html "next page")

next

jax.nn.hard_silu

Contents

- [`hard_sigmoid()`](#jax.nn.hard_sigmoid)

By The JAX authors

© Copyright 2024, The JAX Authors.\
