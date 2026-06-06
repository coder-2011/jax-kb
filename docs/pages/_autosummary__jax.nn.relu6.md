- [](../index.html)
- [API Reference](../jax.html)
- [`jax.nn` module](../jax.nn.html)
- jax.nn.relu6

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.nn.relu6.rst "Download source file")
-  .pdf

# jax.nn.relu6

## Contents

- [`relu6`](#jax.nn.relu6)

# jax.nn.relu6[\#](#jax-nn-relu6 "Link to this heading")

jax.nn.relu6 *= \<jax.\_src.custom_derivatives.custom_jvp object\>*[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/nn/functions.py#L780-L810)[\#](#jax.nn.relu6 "Link to this definition")  
Rectified Linear Unit 6 activation function.

Computes the element-wise function

\\\mathrm{relu6}(x) = \min(\max(x, 0), 6)\\

except under differentiation, we take:

\\\nabla \mathrm{relu}(0) = 0\\

and

\\\nabla \mathrm{relu}(6) = 0\\

Parameters:  
**x** (*ArrayLike*) – input array

Returns:  
An array.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

[`relu()`](jax.nn.relu.html#jax.nn.relu "jax.nn.relu")

[](jax.nn.relu.html "previous page")

previous

jax.nn.relu

[](jax.nn.sigmoid.html "next page")

next

jax.nn.sigmoid

Contents

- [`relu6`](#jax.nn.relu6)

By The JAX authors

© Copyright 2024, The JAX Authors.\
