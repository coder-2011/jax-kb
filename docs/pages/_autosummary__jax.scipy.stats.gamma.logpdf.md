- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.stats.gamma.logpdf

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.stats.gamma.logpdf.rst "Download source file")
-  .pdf

# jax.scipy.stats.gamma.logpdf

## Contents

- [`logpdf()`](#jax.scipy.stats.gamma.logpdf)

# jax.scipy.stats.gamma.logpdf[\#](#jax-scipy-stats-gamma-logpdf "Link to this heading")

jax.scipy.stats.gamma.logpdf(*x*, *a*, *loc=0*, *scale=1*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/stats/gamma.py#L25-L63)[\#](#jax.scipy.stats.gamma.logpdf "Link to this definition")  
Gamma log probability distribution function.

JAX implementation of [`scipy.stats.gamma`](https://scipy.github.io/devdocs/reference/generated/scipy.stats.gamma.html#scipy.stats.gamma "(in SciPy v1.19.0.dev)") `logpdf`.

The Gamma probability distribution is given by

\\f(x, a) = \frac{1}{\Gamma(a)}x^{a-1}e^{-x}\\

Where \\\Gamma(a)\\ is the [`gamma()`](jax.scipy.special.gamma.html#jax.scipy.special.gamma "jax.scipy.special.gamma") function. It is defined for \\x \ge 0\\ and \\a \> 0\\.

Parameters:  
- **x** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, value at which to evaluate the PDF

- **a** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, distribution shape parameter

- **loc** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, distribution offset parameter

- **scale** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, distribution scale parameter

Returns:  
array of logpdf values.

Return type:  
[*Array*](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.scipy.stats.gamma.cdf()`](jax.scipy.stats.gamma.cdf.html#jax.scipy.stats.gamma.cdf "jax.scipy.stats.gamma.cdf")

- [`jax.scipy.stats.gamma.pdf()`](jax.scipy.stats.gamma.pdf.html#jax.scipy.stats.gamma.pdf "jax.scipy.stats.gamma.pdf")

- [`jax.scipy.stats.gamma.sf()`](jax.scipy.stats.gamma.sf.html#jax.scipy.stats.gamma.sf "jax.scipy.stats.gamma.sf")

- [`jax.scipy.stats.gamma.logcdf()`](jax.scipy.stats.gamma.logcdf.html#jax.scipy.stats.gamma.logcdf "jax.scipy.stats.gamma.logcdf")

- [`jax.scipy.stats.gamma.logsf()`](jax.scipy.stats.gamma.logsf.html#jax.scipy.stats.gamma.logsf "jax.scipy.stats.gamma.logsf")

[](jax.scipy.stats.expon.ppf.html "previous page")

previous

jax.scipy.stats.expon.ppf

[](jax.scipy.stats.gamma.pdf.html "next page")

next

jax.scipy.stats.gamma.pdf

Contents

- [`logpdf()`](#jax.scipy.stats.gamma.logpdf)

By The JAX authors

© Copyright 2024, The JAX Authors.\
