- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.stats.multinomial.logpmf

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.stats.multinomial.logpmf.rst "Download source file")
-  .pdf

# jax.scipy.stats.multinomial.logpmf

## Contents

- [`logpmf()`](#jax.scipy.stats.multinomial.logpmf)

# jax.scipy.stats.multinomial.logpmf[\#](#jax-scipy-stats-multinomial-logpmf "Link to this heading")

jax.scipy.stats.multinomial.logpmf(*x*, *n*, *p*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/stats/multinomial.py#L25-L57)[\#](#jax.scipy.stats.multinomial.logpmf "Link to this definition")  
Multinomial log probability mass function.

JAX implementation of [`scipy.stats.multinomial`](https://scipy.github.io/devdocs/reference/generated/scipy.stats.multinomial.html#scipy.stats.multinomial "(in SciPy v1.19.0.dev)") `logpdf`.

The multinomial probability distribution is given by

\\f(x, n, p) = n! \prod\_{i=1}^k \frac{p_i^{x_i}}{x_i!}\\

with \\n = \sum_i x_i\\.

Parameters:  
- **x** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, value at which to evaluate the PMF

- **n** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, distribution shape parameter

- **p** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, distribution shape parameter

Returns:  
array of logpmf values.

Return type:  
[*Array*](jax.Array.html#jax.Array "jax.Array")

See also

[`jax.scipy.stats.multinomial.pmf()`](jax.scipy.stats.multinomial.pmf.html#jax.scipy.stats.multinomial.pmf "jax.scipy.stats.multinomial.pmf")

[](jax.scipy.stats.logistic.sf.html "previous page")

previous

jax.scipy.stats.logistic.sf

[](jax.scipy.stats.multinomial.pmf.html "next page")

next

jax.scipy.stats.multinomial.pmf

Contents

- [`logpmf()`](#jax.scipy.stats.multinomial.logpmf)

By The JAX authors

© Copyright 2024, The JAX Authors.\
