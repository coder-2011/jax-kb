- [](../index.html)
- [API Reference](../jax.html)
- [`jax.nn` module](../jax.nn.html)
- jax.nn.soft_sign

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.nn.soft_sign.rst "Download source file")
-  .pdf

# jax.nn.soft_sign

## Contents

- [`soft_sign()`](#jax.nn.soft_sign)

# jax.nn.soft_sign[\#](#jax-nn-soft-sign "Link to this heading")

jax.nn.soft_sign(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/nn/functions.py#L166-L180)[\#](#jax.nn.soft_sign "Link to this definition")  
Soft-sign activation function.

Computes the element-wise function

\\\mathrm{soft\\sign}(x) = \frac{x}{\|x\| + 1}\\

Parameters:  
**x** (*ArrayLike*) – input array

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.nn.sparse_sigmoid.html "previous page")

previous

jax.nn.sparse_sigmoid

[](jax.nn.silu.html "next page")

next

jax.nn.silu

Contents

- [`soft_sign()`](#jax.nn.soft_sign)

By The JAX authors

© Copyright 2024, The JAX Authors.\
