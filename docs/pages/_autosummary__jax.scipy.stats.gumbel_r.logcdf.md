- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.stats.gumbel_r.logcdf

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.stats.gumbel_r.logcdf.rst "Download source file")
-  .pdf

# jax.scipy.stats.gumbel_r.logcdf

## Contents

- [`logcdf()`](#jax.scipy.stats.gumbel_r.logcdf)

# jax.scipy.stats.gumbel_r.logcdf[\#](#jax-scipy-stats-gumbel-r-logcdf "Link to this heading")

jax.scipy.stats.gumbel_r.logcdf(*x*, *loc=0*, *scale=1*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/stats/gumbel_r.py#L94-L128)[\#](#jax.scipy.stats.gumbel_r.logcdf "Link to this definition")  
Gumbel Distribution (Right Skewed) log cumulative density function.

JAX implementation of [`scipy.stats.gumbel_r`](https://scipy.github.io/devdocs/reference/generated/scipy.stats.gumbel_r.html#scipy.stats.gumbel_r "(in SciPy v1.19.0.dev)") `logcdf`.

\\f\_{cdf}(x; \mu, \beta) = \exp\left( -\exp\left( -\frac{x - \mu}{\beta} \right) \right)\\

Parameters:  
- **x** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – ArrayLike, value at which to evaluate log(cdf)

- **loc** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – ArrayLike, distribution offset (\\\mu\\) (defaulted to 0)

- **scale** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – ArrayLike, distribution scaling (\\\beta\\) (defaulted to 1)

Returns:  
array of logcdf values

Return type:  
[*Array*](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.scipy.stats.gumbel_r.logpdf()`](jax.scipy.stats.gumbel_r.logpdf.html#jax.scipy.stats.gumbel_r.logpdf "jax.scipy.stats.gumbel_r.logpdf")

- [`jax.scipy.stats.gumbel_r.pdf()`](jax.scipy.stats.gumbel_r.pdf.html#jax.scipy.stats.gumbel_r.pdf "jax.scipy.stats.gumbel_r.pdf")

- [`jax.scipy.stats.gumbel_r.cdf()`](jax.scipy.stats.gumbel_r.cdf.html#jax.scipy.stats.gumbel_r.cdf "jax.scipy.stats.gumbel_r.cdf")

- [`jax.scipy.stats.gumbel_r.ppf()`](jax.scipy.stats.gumbel_r.ppf.html#jax.scipy.stats.gumbel_r.ppf "jax.scipy.stats.gumbel_r.ppf")

- [`jax.scipy.stats.gumbel_r.sf()`](jax.scipy.stats.gumbel_r.sf.html#jax.scipy.stats.gumbel_r.sf "jax.scipy.stats.gumbel_r.sf")

- [`jax.scipy.stats.gumbel_r.logsf()`](jax.scipy.stats.gumbel_r.logsf.html#jax.scipy.stats.gumbel_r.logsf "jax.scipy.stats.gumbel_r.logsf")

[](jax.scipy.stats.gumbel_r.cdf.html "previous page")

previous

jax.scipy.stats.gumbel_r.cdf

[](jax.scipy.stats.gumbel_r.sf.html "next page")

next

jax.scipy.stats.gumbel_r.sf

Contents

- [`logcdf()`](#jax.scipy.stats.gumbel_r.logcdf)

By The JAX authors

© Copyright 2024, The JAX Authors.\
