- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.stats.gumbel_l.ppf

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.stats.gumbel_l.ppf.rst "Download source file")
-  .pdf

# jax.scipy.stats.gumbel_l.ppf

## Contents

- [`ppf()`](#jax.scipy.stats.gumbel_l.ppf)

# jax.scipy.stats.gumbel_l.ppf[\#](#jax-scipy-stats-gumbel-l-ppf "Link to this heading")

jax.scipy.stats.gumbel_l.ppf(*p*, *loc=0*, *scale=1*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/stats/gumbel_l.py#L161-L196)[\#](#jax.scipy.stats.gumbel_l.ppf "Link to this definition")  
Gumbel Distribution (Left Skewed) percent point function (inverse of CDF)

JAX implementation of [`scipy.stats.gumbel_l`](https://scipy.github.io/devdocs/reference/generated/scipy.stats.gumbel_l.html#scipy.stats.gumbel_l "(in SciPy v1.19.0.dev)") `ppf`.

\\F\_{ppf}}(p; \mu, \beta) = \mu + \beta \log\left( -\log(1 - p) \right)\\

Parameters:  
- **p** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – ArrayLike, probability value (quantile) at which to evaluate ppf

- **loc** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – ArrayLike, distribution offset (\\\mu\\) (defaulted to 0)

- **scale** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – ArrayLike, distribution scaling (\\\beta\\) (defaulted to 1)

Returns:  
array of ppf values

Return type:  
[*Array*](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.scipy.stats.gumbel_l.logpdf()`](jax.scipy.stats.gumbel_l.logpdf.html#jax.scipy.stats.gumbel_l.logpdf "jax.scipy.stats.gumbel_l.logpdf")

- [`jax.scipy.stats.gumbel_l.pdf()`](jax.scipy.stats.gumbel_l.pdf.html#jax.scipy.stats.gumbel_l.pdf "jax.scipy.stats.gumbel_l.pdf")

- [`jax.scipy.stats.gumbel_l.logcdf()`](jax.scipy.stats.gumbel_l.logcdf.html#jax.scipy.stats.gumbel_l.logcdf "jax.scipy.stats.gumbel_l.logcdf")

- [`jax.scipy.stats.gumbel_l.cdf()`](jax.scipy.stats.gumbel_l.cdf.html#jax.scipy.stats.gumbel_l.cdf "jax.scipy.stats.gumbel_l.cdf")

- [`jax.scipy.stats.gumbel_l.logsf()`](jax.scipy.stats.gumbel_l.logsf.html#jax.scipy.stats.gumbel_l.logsf "jax.scipy.stats.gumbel_l.logsf")

- [`jax.scipy.stats.gumbel_l.sf()`](jax.scipy.stats.gumbel_l.sf.html#jax.scipy.stats.gumbel_l.sf "jax.scipy.stats.gumbel_l.sf")

[](jax.scipy.stats.gumbel_l.logsf.html "previous page")

previous

jax.scipy.stats.gumbel_l.logsf

[](jax.scipy.stats.gumbel_r.logpdf.html "next page")

next

jax.scipy.stats.gumbel_r.logpdf

Contents

- [`ppf()`](#jax.scipy.stats.gumbel_l.ppf)

By The JAX authors

© Copyright 2024, The JAX Authors.\
