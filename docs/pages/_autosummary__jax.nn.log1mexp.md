- [](../index.html)
- [API Reference](../jax.html)
- [`jax.nn` module](../jax.nn.html)
- jax.nn.log1mexp

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.nn.log1mexp.rst "Download source file")
-  .pdf

# jax.nn.log1mexp

## Contents

- [`log1mexp`](#jax.nn.log1mexp)

# jax.nn.log1mexp[\#](#jax-nn-log1mexp "Link to this heading")

jax.nn.log1mexp *= \<jax.\_src.custom_derivatives.custom_jvp object\>*[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/nn/functions.py#L1492-L1512)[\#](#jax.nn.log1mexp "Link to this definition")  
Numerically stable calculation of \\\log(1 - \exp(-x))\\.

This function is undefined for \\x \< 0\\.

Based on [TensorFlow’s implementation](https://www.tensorflow.org/probability/api_docs/python/tfp/math/log1mexp).

References

\[1\]

Martin Mächler. [Accurately Computing log(1 − exp(−\|a\|)) Assessed by the Rmpfr package.](https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf).

Parameters:  
**x** (*ArrayLike*)

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.nn.scaled_dot_general.html "previous page")

previous

jax.nn.scaled_dot_general

[](../jax.ops.html "next page")

next

`jax.ops` module

Contents

- [`log1mexp`](#jax.nn.log1mexp)

By The JAX authors

© Copyright 2024, The JAX Authors.\
