- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.special.gammaincc

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.special.gammaincc.rst "Download source file")
-  .pdf

# jax.scipy.special.gammaincc

## Contents

- [`gammaincc()`](#jax.scipy.special.gammaincc)

# jax.scipy.special.gammaincc[\#](#jax-scipy-special-gammaincc "Link to this heading")

jax.scipy.special.gammaincc(*a*, *x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/special.py#L518-L542)[\#](#jax.scipy.special.gammaincc "Link to this definition")  
The regularized upper incomplete gamma function.

JAX implementation of [`scipy.special.gammaincc`](https://scipy.github.io/devdocs/reference/generated/scipy.special.gammaincc.html#scipy.special.gammaincc "(in SciPy v1.19.0.dev)").

\\\mathrm{gammaincc}(x; a) = \frac{1}{\Gamma(a)}\int_x^\infty t^{a-1}e^{-t}\mathrm{d}t\\

where \\\Gamma(a)\\ is the [`gamma()`](jax.scipy.special.gamma.html#jax.scipy.special.gamma "jax.scipy.special.gamma") function.

Parameters:  
- **a** (*ArrayLike*) – arraylike, real-valued. Positive shape parameter of the gamma distribution.

- **x** (*ArrayLike*) – arraylike, real-valued. Non-negative lower limit of integration

Returns:  
array containing values of the gammaincc function.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.scipy.special.gamma()`](jax.scipy.special.gamma.html#jax.scipy.special.gamma "jax.scipy.special.gamma")

- [`jax.scipy.special.gammainc()`](jax.scipy.special.gammainc.html#jax.scipy.special.gammainc "jax.scipy.special.gammainc")

[](jax.scipy.special.gammainc.html "previous page")

previous

jax.scipy.special.gammainc

[](jax.scipy.special.gammaln.html "next page")

next

jax.scipy.special.gammaln

Contents

- [`gammaincc()`](#jax.scipy.special.gammaincc)

By The JAX authors

© Copyright 2024, The JAX Authors.\
