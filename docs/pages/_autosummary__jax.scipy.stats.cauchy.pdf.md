- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.stats.cauchy.pdf

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.stats.cauchy.pdf.rst "Download source file")
-  .pdf

# jax.scipy.stats.cauchy.pdf

## Contents

- [`pdf()`](#jax.scipy.stats.cauchy.pdf)

# jax.scipy.stats.cauchy.pdf[\#](#jax-scipy-stats-cauchy-pdf "Link to this heading")

jax.scipy.stats.cauchy.pdf(*x*, *loc=0*, *scale=1*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/stats/cauchy.py#L60-L89)[\#](#jax.scipy.stats.cauchy.pdf "Link to this definition")  
Cauchy probability distribution function.

JAX implementation of [`scipy.stats.cauchy`](https://scipy.github.io/devdocs/reference/generated/scipy.stats.cauchy.html#scipy.stats.cauchy "(in SciPy v1.19.0.dev)") `pdf`.

The Cauchy probability distribution function is

\\f(x) = \frac{1}{\pi(1 + x^2)}\\

Parameters:  
- **x** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, value at which to evaluate the PDF

- **loc** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, distribution offset parameter

- **scale** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, distribution scale parameter

Returns:  
array of pdf values

Return type:  
[*Array*](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.scipy.stats.cauchy.cdf()`](jax.scipy.stats.cauchy.cdf.html#jax.scipy.stats.cauchy.cdf "jax.scipy.stats.cauchy.cdf")

- [`jax.scipy.stats.cauchy.sf()`](jax.scipy.stats.cauchy.sf.html#jax.scipy.stats.cauchy.sf "jax.scipy.stats.cauchy.sf")

- [`jax.scipy.stats.cauchy.logcdf()`](jax.scipy.stats.cauchy.logcdf.html#jax.scipy.stats.cauchy.logcdf "jax.scipy.stats.cauchy.logcdf")

- [`jax.scipy.stats.cauchy.logpdf()`](jax.scipy.stats.cauchy.logpdf.html#jax.scipy.stats.cauchy.logpdf "jax.scipy.stats.cauchy.logpdf")

- [`jax.scipy.stats.cauchy.logsf()`](jax.scipy.stats.cauchy.logsf.html#jax.scipy.stats.cauchy.logsf "jax.scipy.stats.cauchy.logsf")

- [`jax.scipy.stats.cauchy.isf()`](jax.scipy.stats.cauchy.isf.html#jax.scipy.stats.cauchy.isf "jax.scipy.stats.cauchy.isf")

- [`jax.scipy.stats.cauchy.ppf()`](jax.scipy.stats.cauchy.ppf.html#jax.scipy.stats.cauchy.ppf "jax.scipy.stats.cauchy.ppf")

[](jax.scipy.stats.cauchy.logpdf.html "previous page")

previous

jax.scipy.stats.cauchy.logpdf

[](jax.scipy.stats.cauchy.cdf.html "next page")

next

jax.scipy.stats.cauchy.cdf

Contents

- [`pdf()`](#jax.scipy.stats.cauchy.pdf)

By The JAX authors

© Copyright 2024, The JAX Authors.\
