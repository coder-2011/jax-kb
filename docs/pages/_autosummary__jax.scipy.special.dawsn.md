- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.special.dawsn

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.special.dawsn.rst "Download source file")
-  .pdf

# jax.scipy.special.dawsn

## Contents

- [`dawsn()`](#jax.scipy.special.dawsn)

# jax.scipy.special.dawsn[\#](#jax-scipy-special-dawsn "Link to this heading")

jax.scipy.special.dawsn(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/special.py#L810-L832)[\#](#jax.scipy.special.dawsn "Link to this definition")  
Dawson’s integral.

JAX implementation of [`scipy.special.dawsn`](https://scipy.github.io/devdocs/reference/generated/scipy.special.dawsn.html#scipy.special.dawsn "(in SciPy v1.19.0.dev)").

\\\mathrm{dawsn}(x) = e^{-x^2} \int_0^x e^{t^2} \\ dt\\

Parameters:  
**x** (*ArrayLike*) – arraylike, real-valued.

Returns:  
array containing values of Dawson’s integral.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.scipy.special.erfcx()`](jax.scipy.special.erfcx.html#jax.scipy.special.erfcx "jax.scipy.special.erfcx")

[](jax.scipy.special.comb.html "previous page")

previous

jax.scipy.special.comb

[](jax.scipy.special.digamma.html "next page")

next

jax.scipy.special.digamma

Contents

- [`dawsn()`](#jax.scipy.special.dawsn)

By The JAX authors

© Copyright 2024, The JAX Authors.\
