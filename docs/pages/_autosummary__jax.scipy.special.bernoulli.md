- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.special.bernoulli

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.special.bernoulli.rst "Download source file")
-  .pdf

# jax.scipy.special.bernoulli

## Contents

- [`bernoulli()`](#jax.scipy.special.bernoulli)

# jax.scipy.special.bernoulli[\#](#jax-scipy-special-bernoulli "Link to this heading")

jax.scipy.special.bernoulli(*n*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/special.py#L3134-L3162)[\#](#jax.scipy.special.bernoulli "Link to this definition")  
Generate the first N Bernoulli numbers.

JAX implementation of [`scipy.special.bernoulli()`](https://scipy.github.io/devdocs/reference/generated/scipy.special.bernoulli.html#scipy.special.bernoulli "(in SciPy v1.19.0.dev)").

Parameters:  
**n** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – integer, the number of Bernoulli terms to generate.

Returns:  
Array containing the first `n` Bernoulli numbers.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Notes

`bernoulli` generates numbers using the \\B_n^-\\ convention, such that \\B_1=-1/2\\.

[](jax.scipy.sparse.linalg.gmres.html "previous page")

previous

jax.scipy.sparse.linalg.gmres

[](jax.scipy.special.beta.html "next page")

next

jax.scipy.special.beta

Contents

- [`bernoulli()`](#jax.scipy.special.bernoulli)

By The JAX authors

© Copyright 2024, The JAX Authors.\
