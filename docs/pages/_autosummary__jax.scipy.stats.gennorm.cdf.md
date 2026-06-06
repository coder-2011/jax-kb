- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.stats.gennorm.cdf

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.stats.gennorm.cdf.rst "Download source file")
-  .pdf

# jax.scipy.stats.gennorm.cdf

## Contents

- [`cdf()`](#jax.scipy.stats.gennorm.cdf)

# jax.scipy.stats.gennorm.cdf[\#](#jax-scipy-stats-gennorm-cdf "Link to this heading")

jax.scipy.stats.gennorm.cdf(*x*, *beta*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/stats/gennorm.py#L49-L76)[\#](#jax.scipy.stats.gennorm.cdf "Link to this definition")  
Generalized normal cumulative distribution function.

JAX implementation of [`scipy.stats.gennorm`](https://scipy.github.io/devdocs/reference/generated/scipy.stats.gennorm.html#scipy.stats.gennorm "(in SciPy v1.19.0.dev)") `cdf`.

The cdf is defined as

\\f\_{cdf}(x, k) = \int\_{-\infty}^x f\_{pdf}(y, k)\mathrm{d}y\\

where \\f\_{pdf}\\ is the probability density function, [`jax.scipy.stats.gennorm.pdf()`](jax.scipy.stats.gennorm.pdf.html#jax.scipy.stats.gennorm.pdf "jax.scipy.stats.gennorm.pdf").

Parameters:  
- **x** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, value at which to evaluate the CDF

- **beta** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, distribution shape parameter

Returns:  
array of cdf values.

Return type:  
[*Array*](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.scipy.stats.gennorm.pdf()`](jax.scipy.stats.gennorm.pdf.html#jax.scipy.stats.gennorm.pdf "jax.scipy.stats.gennorm.pdf")

- [`jax.scipy.stats.gennorm.logpdf()`](jax.scipy.stats.gennorm.logpdf.html#jax.scipy.stats.gennorm.logpdf "jax.scipy.stats.gennorm.logpdf")

[](jax.scipy.stats.gumbel_r.ppf.html "previous page")

previous

jax.scipy.stats.gumbel_r.ppf

[](jax.scipy.stats.gennorm.logpdf.html "next page")

next

jax.scipy.stats.gennorm.logpdf

Contents

- [`cdf()`](#jax.scipy.stats.gennorm.cdf)

By The JAX authors

© Copyright 2024, The JAX Authors.\
