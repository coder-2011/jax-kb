- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.special.gammaln

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.special.gammaln.rst "Download source file")
-  .pdf

# jax.scipy.special.gammaln

## Contents

- [`gammaln()`](#jax.scipy.special.gammaln)

# jax.scipy.special.gammaln[\#](#jax-scipy-special-gammaln "Link to this heading")

jax.scipy.special.gammaln(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/special.py#L43-L70)[\#](#jax.scipy.special.gammaln "Link to this definition")  
Natural log of the absolute value of the gamma function.

JAX implementation of [`scipy.special.gammaln`](https://scipy.github.io/devdocs/reference/generated/scipy.special.gammaln.html#scipy.special.gammaln "(in SciPy v1.19.0.dev)").

\\\mathrm{gammaln}(x) = \log(\|\Gamma(x)\|)\\

Where \\\Gamma\\ is the [`gamma()`](jax.scipy.special.gamma.html#jax.scipy.special.gamma "jax.scipy.special.gamma") function.

Parameters:  
**x** (*ArrayLike*) – arraylike, real valued.

Returns:  
array containing the values of the log-gamma function

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.scipy.special.gammaln()`](#jax.scipy.special.gammaln "jax.scipy.special.gammaln"): the natural log of the gamma function

- [`jax.scipy.special.gammasgn()`](jax.scipy.special.gammasgn.html#jax.scipy.special.gammasgn "jax.scipy.special.gammasgn"): the sign of the gamma function

- [`jax.scipy.special.loggamma()`](jax.scipy.special.loggamma.html#jax.scipy.special.loggamma "jax.scipy.special.loggamma"): the principal branch of the log-gamma function

Notes

`gammaln` does not support complex-valued inputs.

[](jax.scipy.special.gammaincc.html "previous page")

previous

jax.scipy.special.gammaincc

[](jax.scipy.special.gammasgn.html "next page")

next

jax.scipy.special.gammasgn

Contents

- [`gammaln()`](#jax.scipy.special.gammaln)

By The JAX authors

© Copyright 2024, The JAX Authors.\
