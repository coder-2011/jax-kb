- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.stats.wrapcauchy.pdf

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.stats.wrapcauchy.pdf.rst "Download source file")
-  .pdf

# jax.scipy.stats.wrapcauchy.pdf

## Contents

- [`pdf()`](#jax.scipy.stats.wrapcauchy.pdf)

# jax.scipy.stats.wrapcauchy.pdf[\#](#jax-scipy-stats-wrapcauchy-pdf "Link to this heading")

jax.scipy.stats.wrapcauchy.pdf(*x*, *c*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/stats/wrapcauchy.py#L59-L83)[\#](#jax.scipy.stats.wrapcauchy.pdf "Link to this definition")  
Wrapped Cauchy probability distribution function.

JAX implementation of [`scipy.stats.wrapcauchy`](https://scipy.github.io/devdocs/reference/generated/scipy.stats.wrapcauchy.html#scipy.stats.wrapcauchy "(in SciPy v1.19.0.dev)") `pdf`.

The wrapped Cauchy probability distribution function is given by

\\f(x, c) = \frac{1-c^2}{2\pi(1+c^2-2c\cos x)}\\

for \\0\<c\<1\\, and where normalization is on the domain \\0\le x\le 2\pi\\.

Parameters:  
- **x** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, value at which to evaluate the PDF

- **c** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, distribution shape parameter

Returns:  
array of pdf values.

Return type:  
[*Array*](jax.Array.html#jax.Array "jax.Array")

See also

[`jax.scipy.stats.wrapcauchy.logpdf()`](jax.scipy.stats.wrapcauchy.logpdf.html#jax.scipy.stats.wrapcauchy.logpdf "jax.scipy.stats.wrapcauchy.logpdf")

[](jax.scipy.stats.wrapcauchy.logpdf.html "previous page")

previous

jax.scipy.stats.wrapcauchy.logpdf

[](../jax.lax.html "next page")

next

`jax.lax` module

Contents

- [`pdf()`](#jax.scipy.stats.wrapcauchy.pdf)

By The JAX authors

© Copyright 2024, The JAX Authors.\
