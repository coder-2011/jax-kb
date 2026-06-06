- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.stats.pareto.logpdf

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.stats.pareto.logpdf.rst "Download source file")
-  .pdf

# jax.scipy.stats.pareto.logpdf

## Contents

- [`logpdf()`](#jax.scipy.stats.pareto.logpdf)

# jax.scipy.stats.pareto.logpdf[\#](#jax-scipy-stats-pareto-logpdf "Link to this heading")

jax.scipy.stats.pareto.logpdf(*x*, *b*, *loc=0*, *scale=1*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/stats/pareto.py#L25-L68)[\#](#jax.scipy.stats.pareto.logpdf "Link to this definition")  
Pareto log probability distribution function.

JAX implementation of [`scipy.stats.pareto`](https://scipy.github.io/devdocs/reference/generated/scipy.stats.pareto.html#scipy.stats.pareto "(in SciPy v1.19.0.dev)") `logpdf`.

The Pareto probability density function is given by

\\\begin{split}f(x, b) = \begin{cases} bx^{-(b+1)} & x \ge 1\\ 0 & x \< 1 \end{cases}\end{split}\\

and is defined for \\b \> 0\\.

Parameters:  
- **x** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, value at which to evaluate the PDF

- **b** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, distribution shape parameter

- **loc** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, distribution offset parameter

- **scale** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, distribution scale parameter

Returns:  
array of logpdf values.

Return type:  
[*Array*](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.scipy.stats.pareto.logcdf()`](jax.scipy.stats.pareto.logcdf.html#jax.scipy.stats.pareto.logcdf "jax.scipy.stats.pareto.logcdf")

- [`jax.scipy.stats.pareto.logsf()`](jax.scipy.stats.pareto.logsf.html#jax.scipy.stats.pareto.logsf "jax.scipy.stats.pareto.logsf")

- [`jax.scipy.stats.pareto.cdf()`](jax.scipy.stats.pareto.cdf.html#jax.scipy.stats.pareto.cdf "jax.scipy.stats.pareto.cdf")

- [`jax.scipy.stats.pareto.pdf()`](jax.scipy.stats.pareto.pdf.html#jax.scipy.stats.pareto.pdf "jax.scipy.stats.pareto.pdf")

- [`jax.scipy.stats.pareto.ppf()`](jax.scipy.stats.pareto.ppf.html#jax.scipy.stats.pareto.ppf "jax.scipy.stats.pareto.ppf")

- [`jax.scipy.stats.pareto.sf()`](jax.scipy.stats.pareto.sf.html#jax.scipy.stats.pareto.sf "jax.scipy.stats.pareto.sf")

[](jax.scipy.stats.pareto.logcdf.html "previous page")

previous

jax.scipy.stats.pareto.logcdf

[](jax.scipy.stats.pareto.logsf.html "next page")

next

jax.scipy.stats.pareto.logsf

Contents

- [`logpdf()`](#jax.scipy.stats.pareto.logpdf)

By The JAX authors

© Copyright 2024, The JAX Authors.\
