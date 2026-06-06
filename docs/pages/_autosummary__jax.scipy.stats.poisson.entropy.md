- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.stats.poisson.entropy

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.stats.poisson.entropy.rst "Download source file")
-  .pdf

# jax.scipy.stats.poisson.entropy

## Contents

- [`entropy()`](#jax.scipy.stats.poisson.entropy)

# jax.scipy.stats.poisson.entropy[\#](#jax-scipy-stats-poisson-entropy "Link to this heading")

jax.scipy.stats.poisson.entropy(*mu*, *loc=0*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/stats/poisson.py#L122-L207)[\#](#jax.scipy.stats.poisson.entropy "Link to this definition")  
Shannon entropy of the Poisson distribution.

JAX implementation of [`scipy.stats.poisson`](https://scipy.github.io/devdocs/reference/generated/scipy.stats.poisson.html#scipy.stats.poisson "(in SciPy v1.19.0.dev)") `entropy`.

The entropy \\H(X)\\ of a Poisson random variable \\X \sim \text{Poisson}(\mu)\\ is defined as:

\\H(X) = -\sum\_{k=0}^\infty p(k) \log p(k)\\

where \\p(k) = e^{-\mu} \mu^k / k!\\ for \\k \geq \max(0, \lfloor \text{loc} \rfloor)\\.

This implementation uses **regime switching** for numerical stability and performance:

- **Small** \\\mu \< 10\\: Direct summation over PMF with adaptive upper bound \\k \leq \mu + 20\\

- **Medium** \\10 \leq \mu \< 100\\: Summation with bound \\k \leq \mu + 10\sqrt{\mu} + 20\\

- **Large** \\\mu \geq 100\\: Asymptotic Stirling approximation: \\H(\mu) \approx \frac{1}{2} \log(2\pi e \mu) - \frac{1}{12\mu}\\

Matches SciPy to relative error \\\< 10^{-5}\\ across all regimes.

Parameters:  
- **mu** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, mean parameter of the Poisson distribution. Must be `>`` ``0`.

- **loc** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – arraylike, optional location parameter (default: 0). Accepted for API compatibility with scipy but does not affect the entropy

Returns:  
Array of entropy values with shape broadcast from `mu` and `loc`. Returns `NaN` for `mu`` ``<=`` ``0`.

Return type:  
[*Array*](jax.Array.html#jax.Array "jax.Array")

Examples

    >>> from jax.scipy.stats import poisson
    >>> poisson.entropy(5.0)
    Array(2.204394, dtype=float32)
    >>> poisson.entropy(jax.numpy.array([1, 10, 100]))
    Array([1.3048419, 2.561407 , 3.7206903], dtype=float32)

See also

- [`jax.scipy.stats.poisson.cdf()`](jax.scipy.stats.poisson.cdf.html#jax.scipy.stats.poisson.cdf "jax.scipy.stats.poisson.cdf")

- [`jax.scipy.stats.poisson.pmf()`](jax.scipy.stats.poisson.pmf.html#jax.scipy.stats.poisson.pmf "jax.scipy.stats.poisson.pmf")

- [`jax.scipy.stats.poisson.logpmf()`](jax.scipy.stats.poisson.logpmf.html#jax.scipy.stats.poisson.logpmf "jax.scipy.stats.poisson.logpmf")

- [`scipy.stats.poisson`](https://scipy.github.io/devdocs/reference/generated/scipy.stats.poisson.html#scipy.stats.poisson "(in SciPy v1.19.0.dev)")

[](jax.scipy.stats.poisson.cdf.html "previous page")

previous

jax.scipy.stats.poisson.cdf

[](jax.scipy.stats.t.logpdf.html "next page")

next

jax.scipy.stats.t.logpdf

Contents

- [`entropy()`](#jax.scipy.stats.poisson.entropy)

By The JAX authors

© Copyright 2024, The JAX Authors.\
