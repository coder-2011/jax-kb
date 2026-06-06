- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.special.betainc

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.special.betainc.rst "Download source file")
-  .pdf

# jax.scipy.special.betainc

## Contents

- [`betainc()`](#jax.scipy.special.betainc)

# jax.scipy.special.betainc[\#](#jax-scipy-special-betainc "Link to this heading")

jax.scipy.special.betainc(*a*, *b*, *x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/special.py#L435-L462)[\#](#jax.scipy.special.betainc "Link to this definition")  
The regularized incomplete beta function.

JAX implementation of [`scipy.special.betainc`](https://scipy.github.io/devdocs/reference/generated/scipy.special.betainc.html#scipy.special.betainc "(in SciPy v1.19.0.dev)").

\\\mathrm{betainc}(a, b, x) = \frac{1}{B(a, b)}\int_0^x t^{a-1}(1-t)^{b-1}\mathrm{d}t\\

where \\B(a, b)\\ is the [`beta()`](jax.scipy.special.beta.html#jax.scipy.special.beta "jax.scipy.special.beta") function.

Parameters:  
- **a** (*ArrayLike*) – arraylike, real-valued. Parameter *a* of the beta distribution.

- **b** (*ArrayLike*) – arraylike, real-valued. Parameter *b* of the beta distribution.

- **x** (*ArrayLike*) – arraylike, real-valued. Upper limit of the integration.

Returns:  
array containing values of the betainc function

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.scipy.special.beta()`](jax.scipy.special.beta.html#jax.scipy.special.beta "jax.scipy.special.beta")

- [`jax.scipy.special.betaln()`](jax.scipy.special.betaln.html#jax.scipy.special.betaln "jax.scipy.special.betaln")

[](jax.scipy.special.beta.html "previous page")

previous

jax.scipy.special.beta

[](jax.scipy.special.betaln.html "next page")

next

jax.scipy.special.betaln

Contents

- [`betainc()`](#jax.scipy.special.betainc)

By The JAX authors

© Copyright 2024, The JAX Authors.\
