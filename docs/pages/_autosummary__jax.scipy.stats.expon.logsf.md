- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.stats.expon.logsf

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.stats.expon.logsf.rst "Download source file")
-  .pdf

# jax.scipy.stats.expon.logsf

## Contents

- [`logsf()`](#jax.scipy.stats.expon.logsf)

# jax.scipy.stats.expon.logsf[\#](#jax-scipy-stats-expon-logsf "Link to this heading")

jax.scipy.stats.expon.logsf(*x*, *loc=0*, *scale=1*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/stats/expon.py#L169-L203)[\#](#jax.scipy.stats.expon.logsf "Link to this definition")  
Exponential log survival function.

JAX implementation of [`scipy.stats.expon`](https://scipy.github.io/devdocs/reference/generated/scipy.stats.expon.html#scipy.stats.expon "(in SciPy v1.19.0.dev)") `logsf`.

The survival function is defined as

\\f\_{sf}(x) = 1 - f\_{cdf}(x)\\

where \\f\_{cdf}(x)\\ is the exponential cumulative distribution function, [`jax.scipy.stats.expon.cdf()`](jax.scipy.stats.expon.cdf.html#jax.scipy.stats.expon.cdf "jax.scipy.stats.expon.cdf").

Parameters:  
- **x** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, value at which to evaluate the PDF

- **loc** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, distribution offset parameter

- **scale** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, distribution scale parameter

Returns:  
array of pdf values.

Return type:  
[*Array*](jax.Array.html#jax.Array "jax.Array")

See also

[`jax.scipy.stats.expon.cdf()`](jax.scipy.stats.expon.cdf.html#jax.scipy.stats.expon.cdf "jax.scipy.stats.expon.cdf") [`jax.scipy.stats.expon.pdf()`](jax.scipy.stats.expon.pdf.html#jax.scipy.stats.expon.pdf "jax.scipy.stats.expon.pdf") [`jax.scipy.stats.expon.ppf()`](jax.scipy.stats.expon.ppf.html#jax.scipy.stats.expon.ppf "jax.scipy.stats.expon.ppf") [`jax.scipy.stats.expon.sf()`](jax.scipy.stats.expon.sf.html#jax.scipy.stats.expon.sf "jax.scipy.stats.expon.sf") [`jax.scipy.stats.expon.logcdf()`](jax.scipy.stats.expon.logcdf.html#jax.scipy.stats.expon.logcdf "jax.scipy.stats.expon.logcdf") [`jax.scipy.stats.expon.logpdf()`](jax.scipy.stats.expon.logpdf.html#jax.scipy.stats.expon.logpdf "jax.scipy.stats.expon.logpdf") [`jax.scipy.stats.expon.logsf()`](#jax.scipy.stats.expon.logsf "jax.scipy.stats.expon.logsf")

[](jax.scipy.stats.expon.cdf.html "previous page")

previous

jax.scipy.stats.expon.cdf

[](jax.scipy.stats.expon.sf.html "next page")

next

jax.scipy.stats.expon.sf

Contents

- [`logsf()`](#jax.scipy.stats.expon.logsf)

By The JAX authors

© Copyright 2024, The JAX Authors.\
