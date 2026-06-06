- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.special.multigammaln

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.special.multigammaln.rst "Download source file")
-  .pdf

# jax.scipy.special.multigammaln

## Contents

- [`multigammaln()`](#jax.scipy.special.multigammaln)

# jax.scipy.special.multigammaln[\#](#jax-scipy-special-multigammaln "Link to this heading")

jax.scipy.special.multigammaln(*a*, *d*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/special.py#L1109-L1147)[\#](#jax.scipy.special.multigammaln "Link to this definition")  
The natural log of the multivariate gamma function.

JAX implementation of [`scipy.special.multigammaln()`](https://scipy.github.io/devdocs/reference/generated/scipy.special.multigammaln.html#scipy.special.multigammaln "(in SciPy v1.19.0.dev)").

\\\mathrm{multigammaln}(a, d) = \log\Gamma_d(a)\\

where

\\\Gamma_d(a) = \pi^{d(d-1)/4}\prod\_{i=1}^d\Gamma(a-(i-1)/2)\\

and \\\Gamma(x)\\ is the [`gamma()`](jax.scipy.special.gamma.html#jax.scipy.special.gamma "jax.scipy.special.gamma") function.

Parameters:  
- **a** (*ArrayLike*) – arraylike, real-valued.

- **d** (*ArrayLike*) – int, the dimension of the integration space.

Returns:  
array containing values of the log-multigamma function.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.scipy.special.gamma()`](jax.scipy.special.gamma.html#jax.scipy.special.gamma "jax.scipy.special.gamma")

[](jax.scipy.special.lpmn_values.html "previous page")

previous

jax.scipy.special.lpmn_values

[](jax.scipy.special.ndtr.html "next page")

next

jax.scipy.special.ndtr

Contents

- [`multigammaln()`](#jax.scipy.special.multigammaln)

By The JAX authors

© Copyright 2024, The JAX Authors.\
