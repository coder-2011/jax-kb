- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.special.gammainc

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.special.gammainc.rst "Download source file")
-  .pdf

# jax.scipy.special.gammainc

## Contents

- [`gammainc()`](#jax.scipy.special.gammainc)

# jax.scipy.special.gammainc[\#](#jax-scipy-special-gammainc "Link to this heading")

jax.scipy.special.gammainc(*a*, *x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/special.py#L492-L516)[\#](#jax.scipy.special.gammainc "Link to this definition")  
The regularized lower incomplete gamma function.

JAX implementation of [`scipy.special.gammainc`](https://scipy.github.io/devdocs/reference/generated/scipy.special.gammainc.html#scipy.special.gammainc "(in SciPy v1.19.0.dev)").

\\\mathrm{gammainc}(x; a) = \frac{1}{\Gamma(a)}\int_0^x t^{a-1}e^{-t}\mathrm{d}t\\

where \\\Gamma(a)\\ is the [`gamma()`](jax.scipy.special.gamma.html#jax.scipy.special.gamma "jax.scipy.special.gamma") function.

Parameters:  
- **a** (*ArrayLike*) – arraylike, real-valued. Positive shape parameter of the gamma distribution.

- **x** (*ArrayLike*) – arraylike, real-valued. Non-negative upper limit of integration

Returns:  
array containing values of the gammainc function.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.scipy.special.gamma()`](jax.scipy.special.gamma.html#jax.scipy.special.gamma "jax.scipy.special.gamma")

- [`jax.scipy.special.gammaincc()`](jax.scipy.special.gammaincc.html#jax.scipy.special.gammaincc "jax.scipy.special.gammaincc")

[](jax.scipy.special.gamma.html "previous page")

previous

jax.scipy.special.gamma

[](jax.scipy.special.gammaincc.html "next page")

next

jax.scipy.special.gammaincc

Contents

- [`gammainc()`](#jax.scipy.special.gammainc)

By The JAX authors

© Copyright 2024, The JAX Authors.\
