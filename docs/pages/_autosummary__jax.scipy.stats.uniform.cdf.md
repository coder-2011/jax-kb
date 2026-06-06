- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.stats.uniform.cdf

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.stats.uniform.cdf.rst "Download source file")
-  .pdf

# jax.scipy.stats.uniform.cdf

## Contents

- [`cdf()`](#jax.scipy.stats.uniform.cdf)

# jax.scipy.stats.uniform.cdf[\#](#jax-scipy-stats-uniform-cdf "Link to this heading")

jax.scipy.stats.uniform.cdf(*x*, *loc=0*, *scale=1*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/stats/uniform.py#L87-L120)[\#](#jax.scipy.stats.uniform.cdf "Link to this definition")  
Uniform cumulative distribution function.

JAX implementation of [`scipy.stats.uniform`](https://scipy.github.io/devdocs/reference/generated/scipy.stats.uniform.html#scipy.stats.uniform "(in SciPy v1.19.0.dev)") `cdf`.

The cdf is defined as

\\f\_{cdf} = \int\_{-\infty}^x f\_{pdf}(y) \mathrm{d}y\\

where here \\f\_{pdf}\\ is the probability distribution function, [`jax.scipy.stats.uniform.pdf()`](jax.scipy.stats.uniform.pdf.html#jax.scipy.stats.uniform.pdf "jax.scipy.stats.uniform.pdf").

Parameters:  
- **x** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, value at which to evaluate the CDF

- **loc** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, distribution offset parameter

- **scale** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, distribution scale parameter

Returns:  
array of cdf values.

Return type:  
[*Array*](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.scipy.stats.uniform.pdf()`](jax.scipy.stats.uniform.pdf.html#jax.scipy.stats.uniform.pdf "jax.scipy.stats.uniform.pdf")

- [`jax.scipy.stats.uniform.logpdf()`](jax.scipy.stats.uniform.logpdf.html#jax.scipy.stats.uniform.logpdf "jax.scipy.stats.uniform.logpdf")

- [`jax.scipy.stats.uniform.ppf()`](jax.scipy.stats.uniform.ppf.html#jax.scipy.stats.uniform.ppf "jax.scipy.stats.uniform.ppf")

[](jax.scipy.stats.uniform.pdf.html "previous page")

previous

jax.scipy.stats.uniform.pdf

[](jax.scipy.stats.uniform.ppf.html "next page")

next

jax.scipy.stats.uniform.ppf

Contents

- [`cdf()`](#jax.scipy.stats.uniform.cdf)

By The JAX authors

© Copyright 2024, The JAX Authors.\
