- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.stats.truncnorm.sf

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.stats.truncnorm.sf.rst "Download source file")
-  .pdf

# jax.scipy.stats.truncnorm.sf

## Contents

- [`sf()`](#jax.scipy.stats.truncnorm.sf)

# jax.scipy.stats.truncnorm.sf[\#](#jax-scipy-stats-truncnorm-sf "Link to this heading")

jax.scipy.stats.truncnorm.sf(*x*, *a*, *b*, *loc=0*, *scale=1*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/stats/truncnorm.py#L182-L215)[\#](#jax.scipy.stats.truncnorm.sf "Link to this definition")  
Truncated normal distribution survival function.

JAX implementation of [`scipy.stats.truncnorm`](https://scipy.github.io/devdocs/reference/generated/scipy.stats.truncnorm.html#scipy.stats.truncnorm "(in SciPy v1.19.0.dev)") `sf`

The survival function is defined as

\\f\_{sf}(x) = 1 - f\_{cdf}(x)\\

where \\f\_{cdf}(x)\\ is the cumulative distribution function, [`jax.scipy.stats.truncnorm.cdf()`](jax.scipy.stats.truncnorm.cdf.html#jax.scipy.stats.truncnorm.cdf "jax.scipy.stats.truncnorm.cdf").

Parameters:  
- **x** – arraylike, value at which to evaluate the SF

- **a** – arraylike, distribution shape parameter

- **b** – arraylike, distribution shape parameter

- **loc** – arraylike, distribution offset parameter

- **scale** – arraylike, distribution scale parameter

Returns:  
array of sf values.

See also

- [`jax.scipy.stats.truncnorm.cdf()`](jax.scipy.stats.truncnorm.cdf.html#jax.scipy.stats.truncnorm.cdf "jax.scipy.stats.truncnorm.cdf")

- [`jax.scipy.stats.truncnorm.pdf()`](jax.scipy.stats.truncnorm.pdf.html#jax.scipy.stats.truncnorm.pdf "jax.scipy.stats.truncnorm.pdf")

- [`jax.scipy.stats.truncnorm.sf()`](#jax.scipy.stats.truncnorm.sf "jax.scipy.stats.truncnorm.sf")

- [`jax.scipy.stats.truncnorm.logcdf()`](jax.scipy.stats.truncnorm.logcdf.html#jax.scipy.stats.truncnorm.logcdf "jax.scipy.stats.truncnorm.logcdf")

- [`jax.scipy.stats.truncnorm.logpdf()`](jax.scipy.stats.truncnorm.logpdf.html#jax.scipy.stats.truncnorm.logpdf "jax.scipy.stats.truncnorm.logpdf")

[](jax.scipy.stats.truncnorm.pdf.html "previous page")

previous

jax.scipy.stats.truncnorm.pdf

[](jax.scipy.stats.uniform.logpdf.html "next page")

next

jax.scipy.stats.uniform.logpdf

Contents

- [`sf()`](#jax.scipy.stats.truncnorm.sf)

By The JAX authors

© Copyright 2024, The JAX Authors.\
