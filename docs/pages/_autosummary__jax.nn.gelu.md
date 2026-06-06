- [](../index.html)
- [API Reference](../jax.html)
- [`jax.nn` module](../jax.nn.html)
- jax.nn.gelu

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.nn.gelu.rst "Download source file")
-  .pdf

# jax.nn.gelu

## Contents

- [`gelu()`](#jax.nn.gelu)

# jax.nn.gelu[\#](#jax-nn-gelu "Link to this heading")

jax.nn.gelu(*x*, *approximate=True*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/nn/functions.py#L437-L470)[\#](#jax.nn.gelu "Link to this definition")  
Gaussian error linear unit activation function.

If `approximate=False`, computes the element-wise function:

\\\mathrm{gelu}(x) = \frac{x}{2} \left(\mathrm{erfc} \left( \frac{-x}{\sqrt{2}} \right) \right)\\

If `approximate=True`, uses the approximate formulation of GELU:

\\\mathrm{gelu}(x) = \frac{x}{2} \left(1 + \mathrm{tanh} \left( \sqrt{\frac{2}{\pi}} \left(x + 0.044715 x^3 \right) \right) \right)\\

For more information, see [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415), section 2.

Parameters:  
- **x** (*ArrayLike*) – input array

- **approximate** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – whether to use the approximate or exact formulation.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.nn.selu.html "previous page")

previous

jax.nn.selu

[](jax.nn.glu.html "next page")

next

jax.nn.glu

Contents

- [`gelu()`](#jax.nn.gelu)

By The JAX authors

© Copyright 2024, The JAX Authors.\
