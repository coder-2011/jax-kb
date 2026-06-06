- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.stats.beta.logsf

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.stats.beta.logsf.rst "Download source file")
-  .pdf

# jax.scipy.stats.beta.logsf

## Contents

- [`logsf()`](#jax.scipy.stats.beta.logsf)

# jax.scipy.stats.beta.logsf[\#](#jax-scipy-stats-beta-logsf "Link to this heading")

jax.scipy.stats.beta.logsf(*x*, *a*, *b*, *loc=0*, *scale=1*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/stats/beta.py#L230-L263)[\#](#jax.scipy.stats.beta.logsf "Link to this definition")  
Beta distribution log survival function.

JAX implementation of [`scipy.stats.beta`](https://scipy.github.io/devdocs/reference/generated/scipy.stats.beta.html#scipy.stats.beta "(in SciPy v1.19.0.dev)") `logsf`.

The survival function is defined as

\\f\_{sf}(x, a, b) = 1 - f\_{cdf}(x, a, b)\\

where \\f\_{cdf}(x, a, b)\\ is the beta cumulative distribution function, [`jax.scipy.stats.beta.cdf()`](jax.scipy.stats.beta.cdf.html#jax.scipy.stats.beta.cdf "jax.scipy.stats.beta.cdf").

Parameters:  
- **x** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, value at which to evaluate the SF

- **a** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, distribution shape parameter

- **b** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, distribution shape parameter

- **loc** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, distribution offset parameter

- **scale** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, distribution scale parameter

Returns:  
array of logsf values.

Return type:  
[*Array*](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.scipy.stats.beta.cdf()`](jax.scipy.stats.beta.cdf.html#jax.scipy.stats.beta.cdf "jax.scipy.stats.beta.cdf")

- [`jax.scipy.stats.beta.pdf()`](jax.scipy.stats.beta.pdf.html#jax.scipy.stats.beta.pdf "jax.scipy.stats.beta.pdf")

- [`jax.scipy.stats.beta.sf()`](jax.scipy.stats.beta.sf.html#jax.scipy.stats.beta.sf "jax.scipy.stats.beta.sf")

- [`jax.scipy.stats.beta.logcdf()`](jax.scipy.stats.beta.logcdf.html#jax.scipy.stats.beta.logcdf "jax.scipy.stats.beta.logcdf")

- [`jax.scipy.stats.beta.logpdf()`](jax.scipy.stats.beta.logpdf.html#jax.scipy.stats.beta.logpdf "jax.scipy.stats.beta.logpdf")

[](jax.scipy.stats.beta.sf.html "previous page")

previous

jax.scipy.stats.beta.sf

[](jax.scipy.stats.betabinom.logpmf.html "next page")

next

jax.scipy.stats.betabinom.logpmf

Contents

- [`logsf()`](#jax.scipy.stats.beta.logsf)

By The JAX authors

© Copyright 2024, The JAX Authors.\
