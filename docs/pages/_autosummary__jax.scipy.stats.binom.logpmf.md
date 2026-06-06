- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.stats.binom.logpmf

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.stats.binom.logpmf.rst "Download source file")
-  .pdf

# jax.scipy.stats.binom.logpmf

## Contents

- [`logpmf()`](#jax.scipy.stats.binom.logpmf)

# jax.scipy.stats.binom.logpmf[\#](#jax-scipy-stats-binom-logpmf "Link to this heading")

jax.scipy.stats.binom.logpmf(*k*, *n*, *p*, *loc=0*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/stats/binom.py#L25-L63)[\#](#jax.scipy.stats.binom.logpmf "Link to this definition")  
Binomial log probability mass function.

JAX implementation of [`scipy.stats.binom`](https://scipy.github.io/devdocs/reference/generated/scipy.stats.binom.html#scipy.stats.binom "(in SciPy v1.19.0.dev)") `logpmf`.

The binomial probability mass function is defined as

\\f(k, n, p) = {n \choose k}p^k(1-p)^{n-k}\\

for \\0\le p\le 1\\ and non-negative integers \\k\\.

Parameters:  
- **k** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, value at which to evaluate the PMF

- **n** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, distribution shape parameter

- **p** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, distribution shape parameter

- **loc** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, distribution offset parameter

Returns:  
array of logpmf values.

Return type:  
[*Array*](jax.Array.html#jax.Array "jax.Array")

See also

[`jax.scipy.stats.binom.pmf()`](jax.scipy.stats.binom.pmf.html#jax.scipy.stats.binom.pmf "jax.scipy.stats.binom.pmf")

[](jax.scipy.stats.betabinom.pmf.html "previous page")

previous

jax.scipy.stats.betabinom.pmf

[](jax.scipy.stats.binom.pmf.html "next page")

next

jax.scipy.stats.binom.pmf

Contents

- [`logpmf()`](#jax.scipy.stats.binom.logpmf)

By The JAX authors

© Copyright 2024, The JAX Authors.\
