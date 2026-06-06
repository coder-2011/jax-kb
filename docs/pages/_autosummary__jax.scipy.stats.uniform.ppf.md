- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.stats.uniform.ppf

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.stats.uniform.ppf.rst "Download source file")
-  .pdf

# jax.scipy.stats.uniform.ppf

## Contents

- [`ppf()`](#jax.scipy.stats.uniform.ppf)

# jax.scipy.stats.uniform.ppf[\#](#jax-scipy-stats-uniform-ppf "Link to this heading")

jax.scipy.stats.uniform.ppf(*q*, *loc=0*, *scale=1*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/stats/uniform.py#L122-L149)[\#](#jax.scipy.stats.uniform.ppf "Link to this definition")  
Uniform distribution percent point function.

JAX implementation of [`scipy.stats.uniform`](https://scipy.github.io/devdocs/reference/generated/scipy.stats.uniform.html#scipy.stats.uniform "(in SciPy v1.19.0.dev)") `ppf`.

The percent point function is defined as the inverse of the cumulative distribution function, [`jax.scipy.stats.uniform.cdf()`](jax.scipy.stats.uniform.cdf.html#jax.scipy.stats.uniform.cdf "jax.scipy.stats.uniform.cdf").

Parameters:  
- **q** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, value at which to evaluate the PPF

- **loc** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, distribution offset parameter

- **scale** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, distribution scale parameter

Returns:  
array of ppf values.

Return type:  
[*Array*](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.scipy.stats.uniform.cdf()`](jax.scipy.stats.uniform.cdf.html#jax.scipy.stats.uniform.cdf "jax.scipy.stats.uniform.cdf")

- [`jax.scipy.stats.uniform.pdf()`](jax.scipy.stats.uniform.pdf.html#jax.scipy.stats.uniform.pdf "jax.scipy.stats.uniform.pdf")

- [`jax.scipy.stats.uniform.logpdf()`](jax.scipy.stats.uniform.logpdf.html#jax.scipy.stats.uniform.logpdf "jax.scipy.stats.uniform.logpdf")

[](jax.scipy.stats.uniform.cdf.html "previous page")

previous

jax.scipy.stats.uniform.cdf

[](jax.scipy.stats.gaussian_kde.html "next page")

next

jax.scipy.stats.gaussian_kde

Contents

- [`ppf()`](#jax.scipy.stats.uniform.ppf)

By The JAX authors

© Copyright 2024, The JAX Authors.\
