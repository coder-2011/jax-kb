- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.special.polygamma

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.special.polygamma.rst "Download source file")
-  .pdf

# jax.scipy.special.polygamma

## Contents

- [`polygamma()`](#jax.scipy.special.polygamma)

# jax.scipy.special.polygamma[\#](#jax-scipy-special-polygamma "Link to this heading")

jax.scipy.special.polygamma(*n*, *x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/special.py#L1327-L1358)[\#](#jax.scipy.special.polygamma "Link to this definition")  
The polygamma function.

JAX implementation of [`scipy.special.polygamma()`](https://scipy.github.io/devdocs/reference/generated/scipy.special.polygamma.html#scipy.special.polygamma "(in SciPy v1.19.0.dev)").

\\\mathrm{polygamma}(n, x) = \psi^{(n)}(x) = \frac{\mathrm{d}^{n+1}}{\mathrm{d}x^{n+1}} \log \Gamma(x)\\

where \\\psi\\ is the [`digamma()`](jax.scipy.special.digamma.html#jax.scipy.special.digamma "jax.scipy.special.digamma") function and \\\Gamma\\ is the [`gamma()`](jax.scipy.special.gamma.html#jax.scipy.special.gamma "jax.scipy.special.gamma") function.

Parameters:  
- **n** (*ArrayLike*) – arraylike, integer-valued. The order of the derivative.

- **x** (*ArrayLike*) – arraylike, real-valued. The value at which to evaluate the function.

Returns:  
array

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.scipy.special.gamma()`](jax.scipy.special.gamma.html#jax.scipy.special.gamma "jax.scipy.special.gamma")

- [`jax.scipy.special.digamma()`](jax.scipy.special.digamma.html#jax.scipy.special.digamma "jax.scipy.special.digamma")

[](jax.scipy.special.poch.html "previous page")

previous

jax.scipy.special.poch

[](jax.scipy.special.rel_entr.html "next page")

next

jax.scipy.special.rel_entr

Contents

- [`polygamma()`](#jax.scipy.special.polygamma)

By The JAX authors

© Copyright 2024, The JAX Authors.\
