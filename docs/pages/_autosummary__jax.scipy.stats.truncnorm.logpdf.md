- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.stats.truncnorm.logpdf

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.stats.truncnorm.logpdf.rst "Download source file")
-  .pdf

# jax.scipy.stats.truncnorm.logpdf

## Contents

- [`logpdf()`](#jax.scipy.stats.truncnorm.logpdf)

# jax.scipy.stats.truncnorm.logpdf[\#](#jax-scipy-stats-truncnorm-logpdf "Link to this heading")

jax.scipy.stats.truncnorm.logpdf(*x*, *a*, *b*, *loc=0*, *scale=1*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/stats/truncnorm.py#L64-L107)[\#](#jax.scipy.stats.truncnorm.logpdf "Link to this definition")  
Truncated normal log probability distribution function.

JAX implementation of [`scipy.stats.truncnorm`](https://scipy.github.io/devdocs/reference/generated/scipy.stats.truncnorm.html#scipy.stats.truncnorm "(in SciPy v1.19.0.dev)") `logpdf`.

The truncated normal probability distribution is given by

\\\begin{split}f(x, a, b) = \begin{cases} \frac{1}{\sqrt{2\pi}}e^{-x^2/2} & a \le x \le b \\ 0 & \mathrm{otherwise} \end{cases}\end{split}\\

where \\a\\ and \\b\\ are effectively specified in number of standard deviations from zero. JAX uses the scipy nomenclature of `loc` for the centroid and `scale` for the standard deviation.

Parameters:  
- **x** – arraylike, value at which to evaluate the PDF

- **a** – arraylike, distribution shape parameter

- **b** – arraylike, distribution shape parameter

- **loc** – arraylike, distribution offset parameter

- **scale** – arraylike, distribution scale parameter

Returns:  
array of logpdf values.

See also

- [`jax.scipy.stats.truncnorm.cdf()`](jax.scipy.stats.truncnorm.cdf.html#jax.scipy.stats.truncnorm.cdf "jax.scipy.stats.truncnorm.cdf")

- [`jax.scipy.stats.truncnorm.pdf()`](jax.scipy.stats.truncnorm.pdf.html#jax.scipy.stats.truncnorm.pdf "jax.scipy.stats.truncnorm.pdf")

- [`jax.scipy.stats.truncnorm.sf()`](jax.scipy.stats.truncnorm.sf.html#jax.scipy.stats.truncnorm.sf "jax.scipy.stats.truncnorm.sf")

- [`jax.scipy.stats.truncnorm.logcdf()`](jax.scipy.stats.truncnorm.logcdf.html#jax.scipy.stats.truncnorm.logcdf "jax.scipy.stats.truncnorm.logcdf")

- [`jax.scipy.stats.truncnorm.logsf()`](jax.scipy.stats.truncnorm.logsf.html#jax.scipy.stats.truncnorm.logsf "jax.scipy.stats.truncnorm.logsf")

[](jax.scipy.stats.truncnorm.logcdf.html "previous page")

previous

jax.scipy.stats.truncnorm.logcdf

[](jax.scipy.stats.truncnorm.logsf.html "next page")

next

jax.scipy.stats.truncnorm.logsf

Contents

- [`logpdf()`](#jax.scipy.stats.truncnorm.logpdf)

By The JAX authors

© Copyright 2024, The JAX Authors.\
