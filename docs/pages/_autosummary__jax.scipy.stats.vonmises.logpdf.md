- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.stats.vonmises.logpdf

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.stats.vonmises.logpdf.rst "Download source file")
-  .pdf

# jax.scipy.stats.vonmises.logpdf

## Contents

- [`logpdf()`](#jax.scipy.stats.vonmises.logpdf)

# jax.scipy.stats.vonmises.logpdf[\#](#jax-scipy-stats-vonmises-logpdf "Link to this heading")

jax.scipy.stats.vonmises.logpdf(*x*, *kappa*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/stats/vonmises.py#L24-L52)[\#](#jax.scipy.stats.vonmises.logpdf "Link to this definition")  
von Mises log probability distribution function.

JAX implementation of [`scipy.stats.vonmises`](https://scipy.github.io/devdocs/reference/generated/scipy.stats.vonmises.html#scipy.stats.vonmises "(in SciPy v1.19.0.dev)") `logpdf`.

The von Mises probability distribution function is given by

\\f(x, \kappa) = \frac{1}{2\pi I_0(\kappa)}e^{\kappa\cos x}\\

Where \\I_0\\ is the modified Bessel function [`i0()`](jax.scipy.special.i0.html#jax.scipy.special.i0 "jax.scipy.special.i0") and \\\kappa\ge 0\\, and the distribution is normalized in the interval \\-\pi \le x \le \pi\\.

Parameters:  
- **x** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, value at which to evaluate the PDF

- **kappa** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, distribution shape parameter

Returns:  
array of logpdf values.

Return type:  
[*Array*](jax.Array.html#jax.Array "jax.Array")

See also

[`jax.scipy.stats.vonmises.pdf()`](jax.scipy.stats.vonmises.pdf.html#jax.scipy.stats.vonmises.pdf "jax.scipy.stats.vonmises.pdf")

[](jax.scipy.stats.gaussian_kde.logpdf.html "previous page")

previous

jax.scipy.stats.gaussian_kde.logpdf

[](jax.scipy.stats.vonmises.pdf.html "next page")

next

jax.scipy.stats.vonmises.pdf

Contents

- [`logpdf()`](#jax.scipy.stats.vonmises.logpdf)

By The JAX authors

© Copyright 2024, The JAX Authors.\
