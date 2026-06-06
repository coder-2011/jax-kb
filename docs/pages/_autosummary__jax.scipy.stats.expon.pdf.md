- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.stats.expon.pdf

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.stats.expon.pdf.rst "Download source file")
-  .pdf

# jax.scipy.stats.expon.pdf

## Contents

- [`pdf()`](#jax.scipy.stats.expon.pdf)

# jax.scipy.stats.expon.pdf[\#](#jax-scipy-stats-expon-pdf "Link to this heading")

jax.scipy.stats.expon.pdf(*x*, *loc=0*, *scale=1*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/stats/expon.py#L61-L93)[\#](#jax.scipy.stats.expon.pdf "Link to this definition")  
Exponential probability distribution function.

JAX implementation of [`scipy.stats.expon`](https://scipy.github.io/devdocs/reference/generated/scipy.stats.expon.html#scipy.stats.expon "(in SciPy v1.19.0.dev)") `pdf`.

The Exponential probability distribution function is

\\\begin{split}f(x) = \begin{cases} e^{-x} & x \ge 0 \\ 0 & \mathrm{otherwise} \end{cases}\end{split}\\

Parameters:  
- **x** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, value at which to evaluate the PDF

- **loc** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, distribution offset parameter

- **scale** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, distribution scale parameter

Returns:  
array of pdf values.

Return type:  
[*Array*](jax.Array.html#jax.Array "jax.Array")

See also

[`jax.scipy.stats.expon.cdf()`](jax.scipy.stats.expon.cdf.html#jax.scipy.stats.expon.cdf "jax.scipy.stats.expon.cdf") [`jax.scipy.stats.expon.pdf()`](#jax.scipy.stats.expon.pdf "jax.scipy.stats.expon.pdf") [`jax.scipy.stats.expon.ppf()`](jax.scipy.stats.expon.ppf.html#jax.scipy.stats.expon.ppf "jax.scipy.stats.expon.ppf") [`jax.scipy.stats.expon.sf()`](jax.scipy.stats.expon.sf.html#jax.scipy.stats.expon.sf "jax.scipy.stats.expon.sf") [`jax.scipy.stats.expon.logcdf()`](jax.scipy.stats.expon.logcdf.html#jax.scipy.stats.expon.logcdf "jax.scipy.stats.expon.logcdf") [`jax.scipy.stats.expon.logpdf()`](jax.scipy.stats.expon.logpdf.html#jax.scipy.stats.expon.logpdf "jax.scipy.stats.expon.logpdf") [`jax.scipy.stats.expon.logsf()`](jax.scipy.stats.expon.logsf.html#jax.scipy.stats.expon.logsf "jax.scipy.stats.expon.logsf")

[](jax.scipy.stats.expon.logpdf.html "previous page")

previous

jax.scipy.stats.expon.logpdf

[](jax.scipy.stats.expon.logcdf.html "next page")

next

jax.scipy.stats.expon.logcdf

Contents

- [`pdf()`](#jax.scipy.stats.expon.pdf)

By The JAX authors

© Copyright 2024, The JAX Authors.\
