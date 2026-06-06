- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.stats.norm.pdf

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.stats.norm.pdf.rst "Download source file")
-  .pdf

# jax.scipy.stats.norm.pdf

## Contents

- [`pdf()`](#jax.scipy.stats.norm.pdf)

# jax.scipy.stats.norm.pdf[\#](#jax-scipy-stats-norm-pdf "Link to this heading")

jax.scipy.stats.norm.pdf(*x*, *loc=0*, *scale=1*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/stats/norm.py#L60-L89)[\#](#jax.scipy.stats.norm.pdf "Link to this definition")  
Normal probability distribution function.

JAX implementation of [`scipy.stats.norm`](https://scipy.github.io/devdocs/reference/generated/scipy.stats.norm.html#scipy.stats.norm "(in SciPy v1.19.0.dev)") `pdf`.

The normal distribution pdf is given by

\\f(x) = \frac{1}{\sqrt{2\pi}}e^{-x^2/2}\\

Parameters:  
- **x** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, value at which to evaluate the PDF

- **loc** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, distribution offset parameter

- **scale** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, distribution scale parameter

Returns:  
array of pdf values.

Return type:  
[*Array*](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.scipy.stats.norm.cdf()`](jax.scipy.stats.norm.cdf.html#jax.scipy.stats.norm.cdf "jax.scipy.stats.norm.cdf")

- [`jax.scipy.stats.norm.sf()`](jax.scipy.stats.norm.sf.html#jax.scipy.stats.norm.sf "jax.scipy.stats.norm.sf")

- [`jax.scipy.stats.norm.logcdf()`](jax.scipy.stats.norm.logcdf.html#jax.scipy.stats.norm.logcdf "jax.scipy.stats.norm.logcdf")

- [`jax.scipy.stats.norm.logpdf()`](jax.scipy.stats.norm.logpdf.html#jax.scipy.stats.norm.logpdf "jax.scipy.stats.norm.logpdf")

- [`jax.scipy.stats.norm.logsf()`](jax.scipy.stats.norm.logsf.html#jax.scipy.stats.norm.logsf "jax.scipy.stats.norm.logsf")

- [`jax.scipy.stats.norm.isf()`](jax.scipy.stats.norm.isf.html#jax.scipy.stats.norm.isf "jax.scipy.stats.norm.isf")

- [`jax.scipy.stats.norm.ppf()`](jax.scipy.stats.norm.ppf.html#jax.scipy.stats.norm.ppf "jax.scipy.stats.norm.ppf")

[](jax.scipy.stats.norm.logpdf.html "previous page")

previous

jax.scipy.stats.norm.logpdf

[](jax.scipy.stats.norm.cdf.html "next page")

next

jax.scipy.stats.norm.cdf

Contents

- [`pdf()`](#jax.scipy.stats.norm.pdf)

By The JAX authors

© Copyright 2024, The JAX Authors.\
