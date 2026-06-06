- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.stats.truncnorm.logcdf

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.stats.truncnorm.logcdf.rst "Download source file")
-  .pdf

# jax.scipy.stats.truncnorm.logcdf

## Contents

- [`logcdf()`](#jax.scipy.stats.truncnorm.logcdf)

# jax.scipy.stats.truncnorm.logcdf[\#](#jax-scipy-stats-truncnorm-logcdf "Link to this heading")

jax.scipy.stats.truncnorm.logcdf(*x*, *a*, *b*, *loc=0*, *scale=1*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/stats/truncnorm.py#L217-L263)[\#](#jax.scipy.stats.truncnorm.logcdf "Link to this definition")  
Truncated normal log cumulative distribution function.

JAX implementation of [`scipy.stats.truncnorm`](https://scipy.github.io/devdocs/reference/generated/scipy.stats.truncnorm.html#scipy.stats.truncnorm "(in SciPy v1.19.0.dev)") `logcdf`.

The cdf is defined as

\\f\_{cdf} = \int\_{-\infty}^x f\_{pdf}(y) \mathrm{d}y\\

where here \\f\_{pdf}\\ is the probability distribution function, [`jax.scipy.stats.truncnorm.pdf()`](jax.scipy.stats.truncnorm.pdf.html#jax.scipy.stats.truncnorm.pdf "jax.scipy.stats.truncnorm.pdf").

Parameters:  
- **x** – arraylike, value at which to evaluate the CDF

- **a** – arraylike, distribution shape parameter

- **b** – arraylike, distribution shape parameter

- **loc** – arraylike, distribution offset parameter

- **scale** – arraylike, distribution scale parameter

Returns:  
array of logcdf values.

See also

- [`jax.scipy.stats.truncnorm.cdf()`](jax.scipy.stats.truncnorm.cdf.html#jax.scipy.stats.truncnorm.cdf "jax.scipy.stats.truncnorm.cdf")

- [`jax.scipy.stats.truncnorm.pdf()`](jax.scipy.stats.truncnorm.pdf.html#jax.scipy.stats.truncnorm.pdf "jax.scipy.stats.truncnorm.pdf")

- [`jax.scipy.stats.truncnorm.sf()`](jax.scipy.stats.truncnorm.sf.html#jax.scipy.stats.truncnorm.sf "jax.scipy.stats.truncnorm.sf")

- [`jax.scipy.stats.truncnorm.logpdf()`](jax.scipy.stats.truncnorm.logpdf.html#jax.scipy.stats.truncnorm.logpdf "jax.scipy.stats.truncnorm.logpdf")

- [`jax.scipy.stats.truncnorm.logsf()`](jax.scipy.stats.truncnorm.logsf.html#jax.scipy.stats.truncnorm.logsf "jax.scipy.stats.truncnorm.logsf")

[](jax.scipy.stats.truncnorm.cdf.html "previous page")

previous

jax.scipy.stats.truncnorm.cdf

[](jax.scipy.stats.truncnorm.logpdf.html "next page")

next

jax.scipy.stats.truncnorm.logpdf

Contents

- [`logcdf()`](#jax.scipy.stats.truncnorm.logcdf)

By The JAX authors

© Copyright 2024, The JAX Authors.\
