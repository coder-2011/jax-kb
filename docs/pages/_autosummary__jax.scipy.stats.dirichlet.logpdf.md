- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.stats.dirichlet.logpdf

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.stats.dirichlet.logpdf.rst "Download source file")
-  .pdf

# jax.scipy.stats.dirichlet.logpdf

## Contents

- [`logpdf()`](#jax.scipy.stats.dirichlet.logpdf)

# jax.scipy.stats.dirichlet.logpdf[\#](#jax-scipy-stats-dirichlet-logpdf "Link to this heading")

jax.scipy.stats.dirichlet.logpdf(*x*, *alpha*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/stats/dirichlet.py#L30-L55)[\#](#jax.scipy.stats.dirichlet.logpdf "Link to this definition")  
Dirichlet log probability distribution function.

JAX implementation of [`scipy.stats.dirichlet`](https://scipy.github.io/devdocs/reference/generated/scipy.stats.dirichlet.html#scipy.stats.dirichlet "(in SciPy v1.19.0.dev)") `logpdf`.

The Dirichlet probability density function is

\\f(\mathbf{x}) = \frac{1}{B(\mathbf{\alpha})} \prod\_{i=1}^K x_i^{\alpha_i - 1}\\

where \\B(\mathbf{\alpha})\\ is the [`beta()`](jax.scipy.special.beta.html#jax.scipy.special.beta "jax.scipy.special.beta") function in a \\K\\-dimensional vector space.

Parameters:  
- **x** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, value at which to evaluate the PDF

- **alpha** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, distribution shape parameter

Returns:  
array of logpdf values.

Return type:  
[*Array*](jax.Array.html#jax.Array "jax.Array")

See also

[`jax.scipy.stats.dirichlet.pdf()`](jax.scipy.stats.dirichlet.pdf.html#jax.scipy.stats.dirichlet.pdf "jax.scipy.stats.dirichlet.pdf")

[](jax.scipy.stats.chi2.logsf.html "previous page")

previous

jax.scipy.stats.chi2.logsf

[](jax.scipy.stats.dirichlet.pdf.html "next page")

next

jax.scipy.stats.dirichlet.pdf

Contents

- [`logpdf()`](#jax.scipy.stats.dirichlet.logpdf)

By The JAX authors

© Copyright 2024, The JAX Authors.\
