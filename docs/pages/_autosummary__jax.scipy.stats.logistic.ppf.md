- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.stats.logistic.ppf

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.stats.logistic.ppf.rst "Download source file")
-  .pdf

# jax.scipy.stats.logistic.ppf

## Contents

- [`ppf()`](#jax.scipy.stats.logistic.ppf)

# jax.scipy.stats.logistic.ppf[\#](#jax-scipy-stats-logistic-ppf "Link to this heading")

jax.scipy.stats.logistic.ppf(*x*, *loc=0*, *scale=1*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/stats/logistic.py#L87-L112)[\#](#jax.scipy.stats.logistic.ppf "Link to this definition")  
Logistic distribution percent point function.

JAX implementation of [`scipy.stats.logistic`](https://scipy.github.io/devdocs/reference/generated/scipy.stats.logistic.html#scipy.stats.logistic "(in SciPy v1.19.0.dev)") `ppf`.

The percent point function is defined as the inverse of the cumulative distribution function, [`jax.scipy.stats.logistic.cdf()`](jax.scipy.stats.logistic.cdf.html#jax.scipy.stats.logistic.cdf "jax.scipy.stats.logistic.cdf").

Parameters:  
- **x** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, value at which to evaluate the PPF

- **loc** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, distribution offset parameter

- **scale** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, distribution scale parameter

Returns:  
array of ppf values.

Return type:  
[*Array*](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.scipy.stats.logistic.cdf()`](jax.scipy.stats.logistic.cdf.html#jax.scipy.stats.logistic.cdf "jax.scipy.stats.logistic.cdf")

- [`jax.scipy.stats.logistic.pdf()`](jax.scipy.stats.logistic.pdf.html#jax.scipy.stats.logistic.pdf "jax.scipy.stats.logistic.pdf")

- [`jax.scipy.stats.logistic.sf()`](jax.scipy.stats.logistic.sf.html#jax.scipy.stats.logistic.sf "jax.scipy.stats.logistic.sf")

- [`jax.scipy.stats.logistic.isf()`](jax.scipy.stats.logistic.isf.html#jax.scipy.stats.logistic.isf "jax.scipy.stats.logistic.isf")

- [`jax.scipy.stats.logistic.logpdf()`](jax.scipy.stats.logistic.logpdf.html#jax.scipy.stats.logistic.logpdf "jax.scipy.stats.logistic.logpdf")

[](jax.scipy.stats.logistic.pdf.html "previous page")

previous

jax.scipy.stats.logistic.pdf

[](jax.scipy.stats.logistic.sf.html "next page")

next

jax.scipy.stats.logistic.sf

Contents

- [`ppf()`](#jax.scipy.stats.logistic.ppf)

By The JAX authors

© Copyright 2024, The JAX Authors.\
