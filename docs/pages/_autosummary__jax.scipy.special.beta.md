- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.special.beta

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.special.beta.rst "Download source file")
-  .pdf

# jax.scipy.special.beta

## Contents

- [`beta()`](#jax.scipy.special.beta)

# jax.scipy.special.beta[\#](#jax-scipy-special-beta "Link to this heading")

jax.scipy.special.beta(*a*, *b*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/special.py#L406-L433)[\#](#jax.scipy.special.beta "Link to this definition")  
The beta function

JAX implementation of [`scipy.special.beta`](https://scipy.github.io/devdocs/reference/generated/scipy.special.beta.html#scipy.special.beta "(in SciPy v1.19.0.dev)").

\\\mathrm{beta}(a, b) = B(a, b) = \frac{\Gamma(a)\Gamma(b)}{\Gamma(a + b)}\\

where \\\Gamma\\ is the [`gamma()`](jax.scipy.special.gamma.html#jax.scipy.special.gamma "jax.scipy.special.gamma") function.

Parameters:  
- **a** (*ArrayLike*) – arraylike, real-valued. Parameter *a* of the beta distribution.

- **b** (*ArrayLike*) – arraylike, real-valued. Parameter *b* of the beta distribution.

Returns:  
array containing the values of the beta function.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.scipy.special.gamma()`](jax.scipy.special.gamma.html#jax.scipy.special.gamma "jax.scipy.special.gamma")

- [`jax.scipy.special.betaln()`](jax.scipy.special.betaln.html#jax.scipy.special.betaln "jax.scipy.special.betaln")

[](jax.scipy.special.bernoulli.html "previous page")

previous

jax.scipy.special.bernoulli

[](jax.scipy.special.betainc.html "next page")

next

jax.scipy.special.betainc

Contents

- [`beta()`](#jax.scipy.special.beta)

By The JAX authors

© Copyright 2024, The JAX Authors.\
