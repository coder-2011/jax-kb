- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.stats.laplace.pdf

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.stats.laplace.pdf.rst "Download source file")
-  .pdf

# jax.scipy.stats.laplace.pdf

## Contents

- [`pdf()`](#jax.scipy.stats.laplace.pdf)

# jax.scipy.stats.laplace.pdf[\#](#jax-scipy-stats-laplace-pdf "Link to this heading")

jax.scipy.stats.laplace.pdf(*x*, *loc=0*, *scale=1*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/stats/laplace.py#L50-L74)[\#](#jax.scipy.stats.laplace.pdf "Link to this definition")  
Laplace probability distribution function.

JAX implementation of [`scipy.stats.laplace`](https://scipy.github.io/devdocs/reference/generated/scipy.stats.laplace.html#scipy.stats.laplace "(in SciPy v1.19.0.dev)") `pdf`.

The Laplace probability distribution function is given by

\\f(x) = \frac{1}{2} e^{-\|x\|}\\

Parameters:  
- **x** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, value at which to evaluate the PDF

- **loc** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, distribution offset parameter

- **scale** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, distribution scale parameter

Returns:  
array of pdf values.

Return type:  
[*Array*](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.scipy.stats.laplace.cdf()`](jax.scipy.stats.laplace.cdf.html#jax.scipy.stats.laplace.cdf "jax.scipy.stats.laplace.cdf")

- [`jax.scipy.stats.laplace.logpdf()`](jax.scipy.stats.laplace.logpdf.html#jax.scipy.stats.laplace.logpdf "jax.scipy.stats.laplace.logpdf")

[](jax.scipy.stats.laplace.logpdf.html "previous page")

previous

jax.scipy.stats.laplace.logpdf

[](jax.scipy.stats.logistic.cdf.html "next page")

next

jax.scipy.stats.logistic.cdf

Contents

- [`pdf()`](#jax.scipy.stats.laplace.pdf)

By The JAX authors

© Copyright 2024, The JAX Authors.\
