- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.special.gamma

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.special.gamma.rst "Download source file")
-  .pdf

# jax.scipy.special.gamma

## Contents

- [`gamma()`](#jax.scipy.special.gamma)

# jax.scipy.special.gamma[\#](#jax-scipy-special-gamma "Link to this heading")

jax.scipy.special.gamma(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/special.py#L252-L306)[\#](#jax.scipy.special.gamma "Link to this definition")  
The gamma function.

JAX implementation of [`scipy.special.gamma`](https://scipy.github.io/devdocs/reference/generated/scipy.special.gamma.html#scipy.special.gamma "(in SciPy v1.19.0.dev)").

The gamma function is defined for \\\Re(z)\>0\\ as

\\\mathrm{gamma}(z) = \Gamma(z) = \int_0^\infty t^{z-1}e^{-t}\mathrm{d}t\\

and is extended by analytic continuation to arbitrary complex values z. For positive integers n, the gamma function is related to the [`factorial()`](jax.scipy.special.factorial.html#jax.scipy.special.factorial "jax.scipy.special.factorial") function via the following identity:

\\\Gamma(n) = (n - 1)!\\

For real inputs:

- if \\x = -\infty\\, NaN is returned.

- if \\x = \pm 0\\, \\\pm \infty\\ is returned.

- if \\x\\ is a negative integer, NaN is returned. The sign of gamma at a negative integer depends on from which side the pole is approached.

- if \\x = \infty\\, \\\infty\\ is returned.

- if \\x\\ is NaN, NaN is returned.

For complex inputs:

- at non-positive integers (poles), `nan+nanj` is returned, matching SciPy.

- if either real or imaginary component is NaN, `nan+nanj` is returned.

Parameters:  
**x** (*ArrayLike*) – arraylike, real or complex valued. Complex inputs use a Lanczos approximation with reflection formula.

Returns:  
array containing the values of the gamma function. For complex inputs, the output is complex-valued.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.scipy.special.factorial()`](jax.scipy.special.factorial.html#jax.scipy.special.factorial "jax.scipy.special.factorial"): the factorial function.

- [`jax.scipy.special.gammaln()`](jax.scipy.special.gammaln.html#jax.scipy.special.gammaln "jax.scipy.special.gammaln"): the natural log of the gamma function

- [`jax.scipy.special.gammasgn()`](jax.scipy.special.gammasgn.html#jax.scipy.special.gammasgn "jax.scipy.special.gammasgn"): the sign of the gamma function

Notes

For complex inputs, the implementation uses the Lanczos approximation (g=7, N=9 coefficients) with the reflection formula for Re(z) \< 0.5.

[](jax.scipy.special.fresnel.html "previous page")

previous

jax.scipy.special.fresnel

[](jax.scipy.special.gammainc.html "next page")

next

jax.scipy.special.gammainc

Contents

- [`gamma()`](#jax.scipy.special.gamma)

By The JAX authors

© Copyright 2024, The JAX Authors.\
