- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.special.expit

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.special.expit.rst "Download source file")
-  .pdf

# jax.scipy.special.expit

## Contents

- [`expit()`](#jax.scipy.special.expit)

# jax.scipy.special.expit[\#](#jax-scipy-special-expit "Link to this heading")

jax.scipy.special.expit(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/special.py#L915-L932)[\#](#jax.scipy.special.expit "Link to this definition")  
The logistic sigmoid (expit) function

JAX implementation of [`scipy.special.expit`](https://scipy.github.io/devdocs/reference/generated/scipy.special.expit.html#scipy.special.expit "(in SciPy v1.19.0.dev)").

\\\mathrm{expit}(x) = \frac{1}{1 + e^{-x}}\\

Parameters:  
**x** (*ArrayLike*) – arraylike, real-valued.

Returns:  
array containing values of the expit function.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.scipy.special.expi.html "previous page")

previous

jax.scipy.special.expi

[](jax.scipy.special.expn.html "next page")

next

jax.scipy.special.expn

Contents

- [`expit()`](#jax.scipy.special.expit)

By The JAX authors

© Copyright 2024, The JAX Authors.\
