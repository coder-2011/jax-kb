- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.special.expi

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.special.expi.rst "Download source file")
-  .pdf

# jax.scipy.special.expi

## Contents

- [`expi`](#jax.scipy.special.expi)

# jax.scipy.special.expi[\#](#jax-scipy-special-expi "Link to this heading")

jax.scipy.special.expi *= \<jax.\_src.custom_derivatives.custom_jvp object\>*[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/special.py#L2583-L2608)[\#](#jax.scipy.special.expi "Link to this definition")  
Exponential integral function.

JAX implementation of [`scipy.special.expi`](https://scipy.github.io/devdocs/reference/generated/scipy.special.expi.html#scipy.special.expi "(in SciPy v1.19.0.dev)")

\\\mathrm{expi}(x) = \int\_{-\infty}^x \frac{e^t}{t} \mathrm{d}t\\

Parameters:  
**x** (*ArrayLike*) – arraylike, real-valued

Returns:  
array of expi values

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.scipy.special.expn()`](jax.scipy.special.expn.html#jax.scipy.special.expn "jax.scipy.special.expn")

- [`jax.scipy.special.exp1()`](jax.scipy.special.exp1.html#jax.scipy.special.exp1 "jax.scipy.special.exp1")

[](jax.scipy.special.exp1.html "previous page")

previous

jax.scipy.special.exp1

[](jax.scipy.special.expit.html "next page")

next

jax.scipy.special.expit

Contents

- [`expi`](#jax.scipy.special.expi)

By The JAX authors

© Copyright 2024, The JAX Authors.\
