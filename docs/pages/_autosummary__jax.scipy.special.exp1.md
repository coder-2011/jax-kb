- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.special.exp1

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.special.exp1.rst "Download source file")
-  .pdf

# jax.scipy.special.exp1

## Contents

- [`exp1()`](#jax.scipy.special.exp1)

# jax.scipy.special.exp1[\#](#jax-scipy-special-exp1 "Link to this heading")

jax.scipy.special.exp1(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/special.py#L3012-L3036)[\#](#jax.scipy.special.exp1 "Link to this definition")  
Exponential integral function.

JAX implementation of [`scipy.special.exp1`](https://scipy.github.io/devdocs/reference/generated/scipy.special.exp1.html#scipy.special.exp1 "(in SciPy v1.19.0.dev)")

\\\mathrm{exp1}(x) = E_1(x) = x^{n-1}\int_x^\infty\frac{e^{-t}}{t}\mathrm{d}t\\

Parameters:  
**x** (*ArrayLike*) – arraylike, real-valued

Returns:  
array of exp1 values

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.scipy.special.expi()`](jax.scipy.special.expi.html#jax.scipy.special.expi "jax.scipy.special.expi")

- [`jax.scipy.special.expn()`](jax.scipy.special.expn.html#jax.scipy.special.expn "jax.scipy.special.expn")

[](jax.scipy.special.erfinv.html "previous page")

previous

jax.scipy.special.erfinv

[](jax.scipy.special.expi.html "next page")

next

jax.scipy.special.expi

Contents

- [`exp1()`](#jax.scipy.special.exp1)

By The JAX authors

© Copyright 2024, The JAX Authors.\
