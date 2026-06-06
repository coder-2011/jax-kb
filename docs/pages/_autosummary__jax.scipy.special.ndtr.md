- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.special.ndtr

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.special.ndtr.rst "Download source file")
-  .pdf

# jax.scipy.special.ndtr

## Contents

- [`ndtr()`](#jax.scipy.special.ndtr)

# jax.scipy.special.ndtr[\#](#jax-scipy-special-ndtr "Link to this heading")

jax.scipy.special.ndtr(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/special.py#L1437-L1469)[\#](#jax.scipy.special.ndtr "Link to this definition")  
Normal distribution function.

JAX implementation of [`scipy.special.ndtr`](https://scipy.github.io/devdocs/reference/generated/scipy.special.ndtr.html#scipy.special.ndtr "(in SciPy v1.19.0.dev)").

Returns the area under the Gaussian probability density function, integrated from minus infinity to x:

\\\begin{split}\begin{align} \mathrm{ndtr}(x) =& \\ \frac{1}{\sqrt{2 \pi}}\int\_{-\infty}^{x} e^{-\frac{1}{2}t^2} dt \\ =&\\ \frac{1}{2} (1 + \mathrm{erf}(\frac{x}{\sqrt{2}})) \\ =&\\ \frac{1}{2} \mathrm{erfc}(\frac{x}{\sqrt{2}}) \end{align}\end{split}\\

Parameters:  
**x** (*ArrayLike*) – An array of type float32, float64.

Returns:  
An array with dtype=x.dtype.

Raises:  
[**TypeError**](https://docs.python.org/3/library/exceptions.html#TypeError "(in Python v3.14)") – if x is not floating-type.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.scipy.special.multigammaln.html "previous page")

previous

jax.scipy.special.multigammaln

[](jax.scipy.special.ndtri.html "next page")

next

jax.scipy.special.ndtri

Contents

- [`ndtr()`](#jax.scipy.special.ndtr)

By The JAX authors

© Copyright 2024, The JAX Authors.\
