- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.stats.poisson.pmf

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.stats.poisson.pmf.rst "Download source file")
-  .pdf

# jax.scipy.stats.poisson.pmf

## Contents

- [`pmf()`](#jax.scipy.stats.poisson.pmf)

# jax.scipy.stats.poisson.pmf[\#](#jax-scipy-stats-poisson-pmf "Link to this heading")

jax.scipy.stats.poisson.pmf(*k*, *mu*, *loc=0*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/stats/poisson.py#L60-L87)[\#](#jax.scipy.stats.poisson.pmf "Link to this definition")  
Poisson probability mass function.

JAX implementation of [`scipy.stats.poisson`](https://scipy.github.io/devdocs/reference/generated/scipy.stats.poisson.html#scipy.stats.poisson "(in SciPy v1.19.0.dev)") `pmf`.

The Poisson probability mass function is given by

\\f(k) = e^{-\mu}\frac{\mu^k}{k!}\\

and is defined for \\k \ge 0\\ and \\\mu \ge 0\\.

Parameters:  
- **k** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, value at which to evaluate the PMF

- **mu** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, distribution shape parameter

- **loc** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, distribution offset parameter

Returns:  
array of pmf values.

Return type:  
[*Array*](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.scipy.stats.poisson.cdf()`](jax.scipy.stats.poisson.cdf.html#jax.scipy.stats.poisson.cdf "jax.scipy.stats.poisson.cdf")

- [`jax.scipy.stats.poisson.logpmf()`](jax.scipy.stats.poisson.logpmf.html#jax.scipy.stats.poisson.logpmf "jax.scipy.stats.poisson.logpmf")

- [`jax.scipy.stats.poisson.entropy()`](jax.scipy.stats.poisson.entropy.html#jax.scipy.stats.poisson.entropy "jax.scipy.stats.poisson.entropy")

[](jax.scipy.stats.poisson.logpmf.html "previous page")

previous

jax.scipy.stats.poisson.logpmf

[](jax.scipy.stats.poisson.cdf.html "next page")

next

jax.scipy.stats.poisson.cdf

Contents

- [`pmf()`](#jax.scipy.stats.poisson.pmf)

By The JAX authors

© Copyright 2024, The JAX Authors.\
