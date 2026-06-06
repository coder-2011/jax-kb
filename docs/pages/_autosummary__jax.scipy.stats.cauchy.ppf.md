- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.stats.cauchy.ppf

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.stats.cauchy.ppf.rst "Download source file")
-  .pdf

# jax.scipy.stats.cauchy.ppf

## Contents

- [`ppf()`](#jax.scipy.stats.cauchy.ppf)

# jax.scipy.stats.cauchy.ppf[\#](#jax-scipy-stats-cauchy-ppf "Link to this heading")

jax.scipy.stats.cauchy.ppf(*q*, *loc=0*, *scale=1*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/stats/cauchy.py#L264-L294)[\#](#jax.scipy.stats.cauchy.ppf "Link to this definition")  
Cauchy distribution percent point function.

JAX implementation of [`scipy.stats.cauchy`](https://scipy.github.io/devdocs/reference/generated/scipy.stats.cauchy.html#scipy.stats.cauchy "(in SciPy v1.19.0.dev)") `ppf`.

The percent point function is defined as the inverse of the cumulative distribution function, [`jax.scipy.stats.cauchy.cdf()`](jax.scipy.stats.cauchy.cdf.html#jax.scipy.stats.cauchy.cdf "jax.scipy.stats.cauchy.cdf").

Parameters:  
- **q** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, value at which to evaluate the PPF

- **loc** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, distribution offset parameter

- **scale** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, distribution scale parameter

Returns:  
array of ppf values.

Return type:  
[*Array*](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.scipy.stats.cauchy.cdf()`](jax.scipy.stats.cauchy.cdf.html#jax.scipy.stats.cauchy.cdf "jax.scipy.stats.cauchy.cdf")

- [`jax.scipy.stats.cauchy.pdf()`](jax.scipy.stats.cauchy.pdf.html#jax.scipy.stats.cauchy.pdf "jax.scipy.stats.cauchy.pdf")

- [`jax.scipy.stats.cauchy.sf()`](jax.scipy.stats.cauchy.sf.html#jax.scipy.stats.cauchy.sf "jax.scipy.stats.cauchy.sf")

- [`jax.scipy.stats.cauchy.logcdf()`](jax.scipy.stats.cauchy.logcdf.html#jax.scipy.stats.cauchy.logcdf "jax.scipy.stats.cauchy.logcdf")

- [`jax.scipy.stats.cauchy.logpdf()`](jax.scipy.stats.cauchy.logpdf.html#jax.scipy.stats.cauchy.logpdf "jax.scipy.stats.cauchy.logpdf")

- [`jax.scipy.stats.cauchy.logsf()`](jax.scipy.stats.cauchy.logsf.html#jax.scipy.stats.cauchy.logsf "jax.scipy.stats.cauchy.logsf")

- [`jax.scipy.stats.cauchy.isf()`](jax.scipy.stats.cauchy.isf.html#jax.scipy.stats.cauchy.isf "jax.scipy.stats.cauchy.isf")

[](jax.scipy.stats.cauchy.isf.html "previous page")

previous

jax.scipy.stats.cauchy.isf

[](jax.scipy.stats.chi2.logpdf.html "next page")

next

jax.scipy.stats.chi2.logpdf

Contents

- [`ppf()`](#jax.scipy.stats.cauchy.ppf)

By The JAX authors

© Copyright 2024, The JAX Authors.\
