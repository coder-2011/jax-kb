- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.stats.bernoulli.logpmf

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.stats.bernoulli.logpmf.rst "Download source file")
-  .pdf

# jax.scipy.stats.bernoulli.logpmf

## Contents

- [`logpmf()`](#jax.scipy.stats.bernoulli.logpmf)

# jax.scipy.stats.bernoulli.logpmf[\#](#jax-scipy-stats-bernoulli-logpmf "Link to this heading")

jax.scipy.stats.bernoulli.logpmf(*k*, *p*, *loc=0*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/stats/bernoulli.py#L25-L60)[\#](#jax.scipy.stats.bernoulli.logpmf "Link to this definition")  
Bernoulli log probability mass function.

JAX implementation of [`scipy.stats.bernoulli`](https://scipy.github.io/devdocs/reference/generated/scipy.stats.bernoulli.html#scipy.stats.bernoulli "(in SciPy v1.19.0.dev)") `logpmf`

The Bernoulli probability mass function is defined as

\\\begin{split}f(k) = \begin{cases} 1 - p, & k = 0 \\ p, & k = 1 \\ 0, & \mathrm{otherwise} \end{cases}\end{split}\\

Parameters:  
- **k** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, value at which to evaluate the PMF

- **p** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, distribution shape parameter

- **loc** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, distribution offset

Returns:  
array of logpmf values

Return type:  
[*Array*](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.scipy.stats.bernoulli.cdf()`](jax.scipy.stats.bernoulli.cdf.html#jax.scipy.stats.bernoulli.cdf "jax.scipy.stats.bernoulli.cdf")

- [`jax.scipy.stats.bernoulli.pmf()`](jax.scipy.stats.bernoulli.pmf.html#jax.scipy.stats.bernoulli.pmf "jax.scipy.stats.bernoulli.pmf")

- [`jax.scipy.stats.bernoulli.ppf()`](jax.scipy.stats.bernoulli.ppf.html#jax.scipy.stats.bernoulli.ppf "jax.scipy.stats.bernoulli.ppf")

[](jax.scipy.stats.sem.html "previous page")

previous

jax.scipy.stats.sem

[](jax.scipy.stats.bernoulli.pmf.html "next page")

next

jax.scipy.stats.bernoulli.pmf

Contents

- [`logpmf()`](#jax.scipy.stats.bernoulli.logpmf)

By The JAX authors

© Copyright 2024, The JAX Authors.\
