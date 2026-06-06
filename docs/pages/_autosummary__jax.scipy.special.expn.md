- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.special.expn

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.special.expn.rst "Download source file")
-  .pdf

# jax.scipy.special.expn

## Contents

- [`expn`](#jax.scipy.special.expn)

# jax.scipy.special.expn[\#](#jax-scipy-special-expn "Link to this heading")

jax.scipy.special.expn *= \<jax.\_src.custom_derivatives.custom_jvp object\>*[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/special.py#L2952-L3001)[\#](#jax.scipy.special.expn "Link to this definition")  
Generalized exponential integral function.

JAX implementation of [`scipy.special.expn`](https://scipy.github.io/devdocs/reference/generated/scipy.special.expn.html#scipy.special.expn "(in SciPy v1.19.0.dev)").

\\\mathrm{expn}(x) = E_n(x) = x^{n-1}\int_x^\infty\frac{e^{-t}}{t^n}\mathrm{d}t\\

Parameters:  
- **n** (*ArrayLike*) – arraylike, real-valued

- **x** (*ArrayLike*) – arraylike, real-valued

Returns:  
array of expn values

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.scipy.special.expi()`](jax.scipy.special.expi.html#jax.scipy.special.expi "jax.scipy.special.expi")

- [`jax.scipy.special.exp1()`](jax.scipy.special.exp1.html#jax.scipy.special.exp1 "jax.scipy.special.exp1")

[](jax.scipy.special.expit.html "previous page")

previous

jax.scipy.special.expit

[](jax.scipy.special.factorial.html "next page")

next

jax.scipy.special.factorial

Contents

- [`expn`](#jax.scipy.special.expn)

By The JAX authors

© Copyright 2024, The JAX Authors.\
