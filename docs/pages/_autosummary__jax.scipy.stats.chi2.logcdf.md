- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.stats.chi2.logcdf

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.stats.chi2.logcdf.rst "Download source file")
-  .pdf

# jax.scipy.stats.chi2.logcdf

## Contents

- [`logcdf()`](#jax.scipy.stats.chi2.logcdf)

# jax.scipy.stats.chi2.logcdf[\#](#jax-scipy-stats-chi2-logcdf "Link to this heading")

jax.scipy.stats.chi2.logcdf(*x*, *df*, *loc=0*, *scale=1*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/stats/chi2.py#L156-L188)[\#](#jax.scipy.stats.chi2.logcdf "Link to this definition")  
Chi-square log cumulative distribution function.

JAX implementation of [`scipy.stats.chi2`](https://scipy.github.io/devdocs/reference/generated/scipy.stats.chi2.html#scipy.stats.chi2 "(in SciPy v1.19.0.dev)") `logcdf`.

The cdf is defined as

\\f\_{cdf}(x, k) = \int\_{-\infty}^x f\_{pdf}(y, k)\mathrm{d}y\\

where \\f\_{pdf}\\ is the probability density function, [`jax.scipy.stats.chi2.pdf()`](jax.scipy.stats.chi2.pdf.html#jax.scipy.stats.chi2.pdf "jax.scipy.stats.chi2.pdf"). JAX follows the scipy convention of using `df` to denote degrees of freedom.

Parameters:  
- **x** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, value at which to evaluate the CDF

- **df** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, distribution shape parameter

- **loc** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, distribution offset parameter

- **scale** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, distribution scale parameter

Returns:  
array of logcdf values

Return type:  
[*Array*](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.scipy.stats.chi2.cdf()`](jax.scipy.stats.chi2.cdf.html#jax.scipy.stats.chi2.cdf "jax.scipy.stats.chi2.cdf")

- [`jax.scipy.stats.chi2.pdf()`](jax.scipy.stats.chi2.pdf.html#jax.scipy.stats.chi2.pdf "jax.scipy.stats.chi2.pdf")

- [`jax.scipy.stats.chi2.sf()`](jax.scipy.stats.chi2.sf.html#jax.scipy.stats.chi2.sf "jax.scipy.stats.chi2.sf")

- [`jax.scipy.stats.chi2.logpdf()`](jax.scipy.stats.chi2.logpdf.html#jax.scipy.stats.chi2.logpdf "jax.scipy.stats.chi2.logpdf")

- [`jax.scipy.stats.chi2.logsf()`](jax.scipy.stats.chi2.logsf.html#jax.scipy.stats.chi2.logsf "jax.scipy.stats.chi2.logsf")

[](jax.scipy.stats.chi2.cdf.html "previous page")

previous

jax.scipy.stats.chi2.cdf

[](jax.scipy.stats.chi2.sf.html "next page")

next

jax.scipy.stats.chi2.sf

Contents

- [`logcdf()`](#jax.scipy.stats.chi2.logcdf)

By The JAX authors

© Copyright 2024, The JAX Authors.\
