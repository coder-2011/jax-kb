- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.stats.poisson.cdf

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.stats.poisson.cdf.rst "Download source file")
-  .pdf

# jax.scipy.stats.poisson.cdf

## Contents

- [`cdf()`](#jax.scipy.stats.poisson.cdf)

# jax.scipy.stats.poisson.cdf[\#](#jax-scipy-stats-poisson-cdf "Link to this heading")

jax.scipy.stats.poisson.cdf(*k*, *mu*, *loc=0*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/stats/poisson.py#L89-L121)[\#](#jax.scipy.stats.poisson.cdf "Link to this definition")  
Poisson cumulative distribution function.

JAX implementation of [`scipy.stats.poisson`](https://scipy.github.io/devdocs/reference/generated/scipy.stats.poisson.html#scipy.stats.poisson "(in SciPy v1.19.0.dev)") `cdf`.

The cumulative distribution function is defined as:

\\f\_{cdf}(k, \mu) = \sum\_{i=0}^k f\_{pmf}(k, \mu)\\

where \\f\_{pmf}(k, \mu)\\ is the probability mass function [`jax.scipy.stats.poisson.pmf()`](jax.scipy.stats.poisson.pmf.html#jax.scipy.stats.poisson.pmf "jax.scipy.stats.poisson.pmf").

Parameters:  
- **k** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, value at which to evaluate the CDF

- **mu** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, distribution shape parameter

- **loc** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, distribution offset parameter

Returns:  
array of cdf values.

Return type:  
[*Array*](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.scipy.stats.poisson.pmf()`](jax.scipy.stats.poisson.pmf.html#jax.scipy.stats.poisson.pmf "jax.scipy.stats.poisson.pmf")

- [`jax.scipy.stats.poisson.logpmf()`](jax.scipy.stats.poisson.logpmf.html#jax.scipy.stats.poisson.logpmf "jax.scipy.stats.poisson.logpmf")

- [`jax.scipy.stats.poisson.entropy()`](jax.scipy.stats.poisson.entropy.html#jax.scipy.stats.poisson.entropy "jax.scipy.stats.poisson.entropy")

[](jax.scipy.stats.poisson.pmf.html "previous page")

previous

jax.scipy.stats.poisson.pmf

[](jax.scipy.stats.poisson.entropy.html "next page")

next

jax.scipy.stats.poisson.entropy

Contents

- [`cdf()`](#jax.scipy.stats.poisson.cdf)

By The JAX authors

© Copyright 2024, The JAX Authors.\
