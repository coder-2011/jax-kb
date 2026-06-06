- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.stats.bernoulli.ppf

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.stats.bernoulli.ppf.rst "Download source file")
-  .pdf

# jax.scipy.stats.bernoulli.ppf

## Contents

- [`ppf()`](#jax.scipy.stats.bernoulli.ppf)

# jax.scipy.stats.bernoulli.ppf[\#](#jax-scipy-stats-bernoulli-ppf "Link to this heading")

jax.scipy.stats.bernoulli.ppf(*q*, *p*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/stats/bernoulli.py#L131-L158)[\#](#jax.scipy.stats.bernoulli.ppf "Link to this definition")  
Bernoulli percent point function.

JAX implementation of [`scipy.stats.bernoulli`](https://scipy.github.io/devdocs/reference/generated/scipy.stats.bernoulli.html#scipy.stats.bernoulli "(in SciPy v1.19.0.dev)") `ppf`

The percent point function is the inverse of the cumulative distribution function, [`jax.scipy.stats.bernoulli.cdf()`](jax.scipy.stats.bernoulli.cdf.html#jax.scipy.stats.bernoulli.cdf "jax.scipy.stats.bernoulli.cdf").

Parameters:  
- **q** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, value at which to evaluate the PPF

- **p** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, distribution shape parameter

Returns:  
array of ppf values

Return type:  
[*Array*](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.scipy.stats.bernoulli.cdf()`](jax.scipy.stats.bernoulli.cdf.html#jax.scipy.stats.bernoulli.cdf "jax.scipy.stats.bernoulli.cdf")

- [`jax.scipy.stats.bernoulli.logpmf()`](jax.scipy.stats.bernoulli.logpmf.html#jax.scipy.stats.bernoulli.logpmf "jax.scipy.stats.bernoulli.logpmf")

- [`jax.scipy.stats.bernoulli.pmf()`](jax.scipy.stats.bernoulli.pmf.html#jax.scipy.stats.bernoulli.pmf "jax.scipy.stats.bernoulli.pmf")

[](jax.scipy.stats.bernoulli.cdf.html "previous page")

previous

jax.scipy.stats.bernoulli.cdf

[](jax.scipy.stats.beta.logpdf.html "next page")

next

jax.scipy.stats.beta.logpdf

Contents

- [`ppf()`](#jax.scipy.stats.bernoulli.ppf)

By The JAX authors

© Copyright 2024, The JAX Authors.\
