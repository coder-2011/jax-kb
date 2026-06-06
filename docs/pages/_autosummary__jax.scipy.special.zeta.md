- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.special.zeta

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.special.zeta.rst "Download source file")
-  .pdf

# jax.scipy.special.zeta

## Contents

- [`zeta`](#jax.scipy.special.zeta)

# jax.scipy.special.zeta[\#](#jax-scipy-special-zeta "Link to this heading")

jax.scipy.special.zeta *= \<jax.\_src.custom_derivatives.custom_jvp object\>*[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/special.py#L1268-L1291)[\#](#jax.scipy.special.zeta "Link to this definition")  
The Hurwitz zeta function.

JAX implementation of [`scipy.special.zeta()`](https://scipy.github.io/devdocs/reference/generated/scipy.special.zeta.html#scipy.special.zeta "(in SciPy v1.19.0.dev)"). JAX does not implement the Riemann zeta function (i.e. `q`` ``=`` ``None`).

\\\zeta(x, q) = \sum\_{n=0}^\infty \frac{1}{(n + q)^x}\\

Parameters:  
- **x** (*ArrayLike*) – arraylike, real-valued

- **q** (*ArrayLike* *\|* *None*) – arraylike, real-valued

Returns:  
array of zeta function values

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.scipy.special.xlogy.html "previous page")

previous

jax.scipy.special.xlogy

[](jax.scipy.stats.mode.html "next page")

next

jax.scipy.stats.mode

Contents

- [`zeta`](#jax.scipy.special.zeta)

By The JAX authors

© Copyright 2024, The JAX Authors.\
