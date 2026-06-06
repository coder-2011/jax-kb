- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.special.loggamma

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.special.loggamma.rst "Download source file")
-  .pdf

# jax.scipy.special.loggamma

## Contents

- [`loggamma()`](#jax.scipy.special.loggamma)

# jax.scipy.special.loggamma[\#](#jax-scipy-special-loggamma "Link to this heading")

jax.scipy.special.loggamma(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/special.py#L225-L250)[\#](#jax.scipy.special.loggamma "Link to this definition")  
Principal branch of the logarithm of the gamma function.

JAX implementation of [`scipy.special.loggamma`](https://scipy.github.io/devdocs/reference/generated/scipy.special.loggamma.html#scipy.special.loggamma "(in SciPy v1.19.0.dev)").

Defined to be \\\log(\Gamma(x))\\ for \\x \> 0\\ and extended to the complex plane by analytic continuation. The function has a single branch cut on the negative real axis.

Parameters:  
**x** (*ArrayLike*) – arraylike, real or complex valued.

Returns:  
array containing the values of the loggamma function. For complex inputs, the output is complex-valued.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.scipy.special.gamma()`](jax.scipy.special.gamma.html#jax.scipy.special.gamma "jax.scipy.special.gamma"): the gamma function.

- [`jax.scipy.special.gammaln()`](jax.scipy.special.gammaln.html#jax.scipy.special.gammaln "jax.scipy.special.gammaln"): the natural log of the absolute value of the gamma function.

[](jax.scipy.special.logit.html "previous page")

previous

jax.scipy.special.logit

[](jax.scipy.special.logsumexp.html "next page")

next

jax.scipy.special.logsumexp

Contents

- [`loggamma()`](#jax.scipy.special.loggamma)

By The JAX authors

© Copyright 2024, The JAX Authors.\
