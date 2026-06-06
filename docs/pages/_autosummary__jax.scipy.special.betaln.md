- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.special.betaln

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.special.betaln.rst "Download source file")
-  .pdf

# jax.scipy.special.betaln

## Contents

- [`betaln()`](#jax.scipy.special.betaln)

# jax.scipy.special.betaln[\#](#jax-scipy-special-betaln "Link to this heading")

jax.scipy.special.betaln(*a*, *b*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/special.py#L308-L331)[\#](#jax.scipy.special.betaln "Link to this definition")  
Natural log of the absolute value of the beta function

JAX implementation of [`scipy.special.betaln`](https://scipy.github.io/devdocs/reference/generated/scipy.special.betaln.html#scipy.special.betaln "(in SciPy v1.19.0.dev)").

\\\mathrm{betaln}(a, b) = \log B(a, b)\\

where \\B\\ is the [`beta()`](jax.scipy.special.beta.html#jax.scipy.special.beta "jax.scipy.special.beta") function.

Parameters:  
- **a** (*ArrayLike*) – arraylike, real-valued. Parameter *a* of the beta distribution.

- **b** (*ArrayLike*) – arraylike, real-valued. Parameter *b* of the beta distribution.

Returns:  
array containing the values of the log-beta function

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

[`jax.scipy.special.beta()`](jax.scipy.special.beta.html#jax.scipy.special.beta "jax.scipy.special.beta")

[](jax.scipy.special.betainc.html "previous page")

previous

jax.scipy.special.betainc

[](jax.scipy.special.boxcox.html "next page")

next

jax.scipy.special.boxcox

Contents

- [`betaln()`](#jax.scipy.special.betaln)

By The JAX authors

© Copyright 2024, The JAX Authors.\
