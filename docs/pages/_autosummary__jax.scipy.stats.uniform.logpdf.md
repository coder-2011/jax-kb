- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.stats.uniform.logpdf

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.stats.uniform.logpdf.rst "Download source file")
-  .pdf

# jax.scipy.stats.uniform.logpdf

## Contents

- [`logpdf()`](#jax.scipy.stats.uniform.logpdf)

# jax.scipy.stats.uniform.logpdf[\#](#jax-scipy-stats-uniform-logpdf "Link to this heading")

jax.scipy.stats.uniform.logpdf(*x*, *loc=0*, *scale=1*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/stats/uniform.py#L23-L55)[\#](#jax.scipy.stats.uniform.logpdf "Link to this definition")  
Uniform log probability distribution function.

JAX implementation of [`scipy.stats.uniform`](https://scipy.github.io/devdocs/reference/generated/scipy.stats.uniform.html#scipy.stats.uniform "(in SciPy v1.19.0.dev)") `logpdf`.

The uniform distribution pdf is given by

\\\begin{split}f(x) = \begin{cases} 1 & 0 \le x \le 1 \\ 0 & \mathrm{otherwise} \end{cases}\end{split}\\

Parameters:  
- **x** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, value at which to evaluate the PDF

- **loc** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, distribution offset parameter

- **scale** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, distribution scale parameter

Returns:  
array of logpdf values

Return type:  
[*Array*](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.scipy.stats.uniform.cdf()`](jax.scipy.stats.uniform.cdf.html#jax.scipy.stats.uniform.cdf "jax.scipy.stats.uniform.cdf")

- [`jax.scipy.stats.uniform.pdf()`](jax.scipy.stats.uniform.pdf.html#jax.scipy.stats.uniform.pdf "jax.scipy.stats.uniform.pdf")

- [`jax.scipy.stats.uniform.ppf()`](jax.scipy.stats.uniform.ppf.html#jax.scipy.stats.uniform.ppf "jax.scipy.stats.uniform.ppf")

[](jax.scipy.stats.truncnorm.sf.html "previous page")

previous

jax.scipy.stats.truncnorm.sf

[](jax.scipy.stats.uniform.pdf.html "next page")

next

jax.scipy.stats.uniform.pdf

Contents

- [`logpdf()`](#jax.scipy.stats.uniform.logpdf)

By The JAX authors

© Copyright 2024, The JAX Authors.\
