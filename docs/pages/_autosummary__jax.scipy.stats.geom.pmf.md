- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.stats.geom.pmf

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.stats.geom.pmf.rst "Download source file")
-  .pdf

# jax.scipy.stats.geom.pmf

## Contents

- [`pmf()`](#jax.scipy.stats.geom.pmf)

# jax.scipy.stats.geom.pmf[\#](#jax-scipy-stats-geom-pmf "Link to this heading")

jax.scipy.stats.geom.pmf(*k*, *p*, *loc=0*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/stats/geom.py#L57-L82)[\#](#jax.scipy.stats.geom.pmf "Link to this definition")  
Geometric probability mass function.

JAX implementation of [`scipy.stats.geom`](https://scipy.github.io/devdocs/reference/generated/scipy.stats.geom.html#scipy.stats.geom "(in SciPy v1.19.0.dev)") `pmf`.

The Geometric probability mass function is given by

\\f(k) = (1 - p)^{k-1}p\\

for \\k\ge 1\\ and \\0 \le p \le 1\\.

Parameters:  
- **k** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, value at which to evaluate the PMF

- **p** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, distribution shape parameter

- **loc** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, distribution offset parameter

Returns:  
array of pmf values.

Return type:  
[*Array*](jax.Array.html#jax.Array "jax.Array")

See also

[`jax.scipy.stats.geom.logpmf()`](jax.scipy.stats.geom.logpmf.html#jax.scipy.stats.geom.logpmf "jax.scipy.stats.geom.logpmf")

[](jax.scipy.stats.geom.logpmf.html "previous page")

previous

jax.scipy.stats.geom.logpmf

[](jax.scipy.stats.laplace.cdf.html "next page")

next

jax.scipy.stats.laplace.cdf

Contents

- [`pmf()`](#jax.scipy.stats.geom.pmf)

By The JAX authors

© Copyright 2024, The JAX Authors.\
