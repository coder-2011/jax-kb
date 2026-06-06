- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.stats.multivariate_normal.pdf

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.stats.multivariate_normal.pdf.rst "Download source file")
-  .pdf

# jax.scipy.stats.multivariate_normal.pdf

## Contents

- [`pdf()`](#jax.scipy.stats.multivariate_normal.pdf)

# jax.scipy.stats.multivariate_normal.pdf[\#](#jax-scipy-stats-multivariate-normal-pdf "Link to this heading")

jax.scipy.stats.multivariate_normal.pdf(*x*, *mean*, *cov*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/stats/multivariate_normal.py#L77-L103)[\#](#jax.scipy.stats.multivariate_normal.pdf "Link to this definition")  
Multivariate normal probability distribution function.

JAX implementation of [`scipy.stats.multivariate_normal`](https://scipy.github.io/devdocs/reference/generated/scipy.stats.multivariate_normal.html#scipy.stats.multivariate_normal "(in SciPy v1.19.0.dev)") `pdf`.

The multivariate normal PDF is defined as

\\f(x) = \frac{1}{(2\pi)^k\det\Sigma}\exp\left(-\frac{(x-\mu)^T\Sigma^{-1}(x-\mu)}{2} \right)\\

where \\\mu\\ is the `mean`, \\\Sigma\\ is the covariance matrix (`cov`), and \\k\\ is the rank of \\\Sigma\\.

Parameters:  
- **x** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, value at which to evaluate the PDF

- **mean** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, centroid of distribution

- **cov** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, covariance matrix of distribution

Returns:  
array of pdf values.

Return type:  
[*Array*](jax.Array.html#jax.Array "jax.Array")

See also

[`jax.scipy.stats.multivariate_normal.logpdf()`](jax.scipy.stats.multivariate_normal.logpdf.html#jax.scipy.stats.multivariate_normal.logpdf "jax.scipy.stats.multivariate_normal.logpdf")

[](jax.scipy.stats.multivariate_normal.logpdf.html "previous page")

previous

jax.scipy.stats.multivariate_normal.logpdf

[](jax.scipy.stats.nbinom.logpmf.html "next page")

next

jax.scipy.stats.nbinom.logpmf

Contents

- [`pdf()`](#jax.scipy.stats.multivariate_normal.pdf)

By The JAX authors

© Copyright 2024, The JAX Authors.\
