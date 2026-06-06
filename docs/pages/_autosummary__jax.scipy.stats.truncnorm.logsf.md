- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.stats.truncnorm.logsf

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.stats.truncnorm.logsf.rst "Download source file")
-  .pdf

# jax.scipy.stats.truncnorm.logsf

## Contents

- [`logsf()`](#jax.scipy.stats.truncnorm.logsf)

# jax.scipy.stats.truncnorm.logsf[\#](#jax-scipy-stats-truncnorm-logsf "Link to this heading")

jax.scipy.stats.truncnorm.logsf(*x*, *a*, *b*, *loc=0*, *scale=1*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/stats/truncnorm.py#L146-L180)[\#](#jax.scipy.stats.truncnorm.logsf "Link to this definition")  
Truncated normal distribution log survival function.

JAX implementation of [`scipy.stats.truncnorm`](https://scipy.github.io/devdocs/reference/generated/scipy.stats.truncnorm.html#scipy.stats.truncnorm "(in SciPy v1.19.0.dev)") `logsf`

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
array of logsf values.

See also

- [`jax.scipy.stats.truncnorm.cdf()`](jax.scipy.stats.truncnorm.cdf.html#jax.scipy.stats.truncnorm.cdf "jax.scipy.stats.truncnorm.cdf")

- [`jax.scipy.stats.truncnorm.pdf()`](jax.scipy.stats.truncnorm.pdf.html#jax.scipy.stats.truncnorm.pdf "jax.scipy.stats.truncnorm.pdf")

- [`jax.scipy.stats.truncnorm.sf()`](jax.scipy.stats.truncnorm.sf.html#jax.scipy.stats.truncnorm.sf "jax.scipy.stats.truncnorm.sf")

- [`jax.scipy.stats.truncnorm.logcdf()`](jax.scipy.stats.truncnorm.logcdf.html#jax.scipy.stats.truncnorm.logcdf "jax.scipy.stats.truncnorm.logcdf")

- [`jax.scipy.stats.truncnorm.logpdf()`](jax.scipy.stats.truncnorm.logpdf.html#jax.scipy.stats.truncnorm.logpdf "jax.scipy.stats.truncnorm.logpdf")

[](jax.scipy.stats.truncnorm.logpdf.html "previous page")

previous

jax.scipy.stats.truncnorm.logpdf

[](jax.scipy.stats.truncnorm.pdf.html "next page")

next

jax.scipy.stats.truncnorm.pdf

Contents

- [`logsf()`](#jax.scipy.stats.truncnorm.logsf)

By The JAX authors

© Copyright 2024, The JAX Authors.\
