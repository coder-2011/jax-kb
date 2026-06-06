- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.special.gammasgn

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.special.gammasgn.rst "Download source file")
-  .pdf

# jax.scipy.special.gammasgn

## Contents

- [`gammasgn()`](#jax.scipy.special.gammasgn)

# jax.scipy.special.gammasgn[\#](#jax-scipy-special-gammasgn "Link to this heading")

jax.scipy.special.gammasgn(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/special.py#L72-L116)[\#](#jax.scipy.special.gammasgn "Link to this definition")  
Sign of the gamma function.

JAX implementation of [`scipy.special.gammasgn`](https://scipy.github.io/devdocs/reference/generated/scipy.special.gammasgn.html#scipy.special.gammasgn "(in SciPy v1.19.0.dev)").

\\\begin{split}\mathrm{gammasgn}(x) = \begin{cases} +1 & \Gamma(x) \> 0 \\ -1 & \Gamma(x) \< 0 \end{cases}\end{split}\\

Where \\\Gamma\\ is the [`gamma()`](jax.scipy.special.gamma.html#jax.scipy.special.gamma "jax.scipy.special.gamma") function. Because \\\Gamma(x)\\ is never zero, no condition is required for this case.

- if \\x = -\infty\\, NaN is returned.

- if \\x = \pm 0\\, \\\pm 1\\ is returned.

- if \\x\\ is a negative integer, NaN is returned. The sign of gamma at a negative integer depends on from which side the pole is approached.

- if \\x = \infty\\, \\1\\ is returned.

- if \\x\\ is NaN, NaN is returned.

Parameters:  
**x** (*ArrayLike*) – arraylike, real valued.

Returns:  
array containing the sign of the gamma function

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.scipy.special.gamma()`](jax.scipy.special.gamma.html#jax.scipy.special.gamma "jax.scipy.special.gamma"): the gamma function

- [`jax.scipy.special.gammaln()`](jax.scipy.special.gammaln.html#jax.scipy.special.gammaln "jax.scipy.special.gammaln"): the natural log of the gamma function

[](jax.scipy.special.gammaln.html "previous page")

previous

jax.scipy.special.gammaln

[](jax.scipy.special.hyp1f1.html "next page")

next

jax.scipy.special.hyp1f1

Contents

- [`gammasgn()`](#jax.scipy.special.gammasgn)

By The JAX authors

© Copyright 2024, The JAX Authors.\
