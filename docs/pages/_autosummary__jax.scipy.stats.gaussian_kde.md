- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.stats.gaussian_kde

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.stats.gaussian_kde.rst "Download source file")
-  .pdf

# jax.scipy.stats.gaussian_kde

## Contents

- [`gaussian_kde`](#jax.scipy.stats.gaussian_kde)
  - [`gaussian_kde.__init__()`](#jax.scipy.stats.gaussian_kde.__init__)

# jax.scipy.stats.gaussian_kde[\#](#jax-scipy-stats-gaussian-kde "Link to this heading")

*class* jax.scipy.stats.gaussian_kde(*dataset*, *bw_method=None*, *weights=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/stats/kde.py#L34-L246)[\#](#jax.scipy.stats.gaussian_kde "Link to this definition")  
Gaussian Kernel Density Estimator

JAX implementation of [`scipy.stats.gaussian_kde`](https://scipy.github.io/devdocs/reference/generated/scipy.stats.gaussian_kde.html#scipy.stats.gaussian_kde "(in SciPy v1.19.0.dev)").

Parameters:  
- **dataset** ([*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.14)")) – arraylike, real-valued. Data from which to estimate the distribution. If 1D, shape is (n_data,). If 2D, shape is (n_dimensions, n_data).

- **bw_method** (*None* *\|* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* [*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "collections.abc.Callable")*\[\[*[*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.14)")*\],* [*Array*](jax.Array.html#jax.Array "jax.Array")*\]*) – string, scalar, or callable. Either “scott”, “silverman”, a scalar value, or a callable function which takes `self` as a parameter.

- **weights** ([*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.14)")) – arraylike, optional. Weights of the same shape as the dataset.

\_\_init\_\_(*dataset*, *bw_method=None*, *weights=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/stats/kde.py#L54-L101)[\#](#jax.scipy.stats.gaussian_kde.__init__ "Link to this definition")  
Parameters:  
**bw_method** (*None* *\|* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* [*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "collections.abc.Callable")*\[\[*[*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.14)")*\],* [*Array*](jax.Array.html#jax.Array "jax.Array")*\]*)

Methods

|  |  |
|----|----|
| [`__init__`](#jax.scipy.stats.gaussian_kde.__init__ "jax.scipy.stats.gaussian_kde.__init__")(dataset\[, bw_method, weights\]) |  |
| [`evaluate`](jax.scipy.stats.gaussian_kde.evaluate.html#jax.scipy.stats.gaussian_kde.evaluate "jax.scipy.stats.gaussian_kde.evaluate")(points) | Evaluate the Gaussian KDE on the given points. |
| `integrate_box`(low_bounds, high_bounds\[, maxpts\]) | This method is not implemented in the JAX interface. |
| [`integrate_box_1d`](jax.scipy.stats.gaussian_kde.integrate_box_1d.html#jax.scipy.stats.gaussian_kde.integrate_box_1d "jax.scipy.stats.gaussian_kde.integrate_box_1d")(low, high) | Integrate the distribution over the given limits. |
| [`integrate_gaussian`](jax.scipy.stats.gaussian_kde.integrate_gaussian.html#jax.scipy.stats.gaussian_kde.integrate_gaussian "jax.scipy.stats.gaussian_kde.integrate_gaussian")(mean, cov) | Integrate the distribution weighted by a Gaussian. |
| [`integrate_kde`](jax.scipy.stats.gaussian_kde.integrate_kde.html#jax.scipy.stats.gaussian_kde.integrate_kde "jax.scipy.stats.gaussian_kde.integrate_kde")(other) | Integrate the product of two Gaussian KDE distributions. |
| [`logpdf`](jax.scipy.stats.gaussian_kde.logpdf.html#jax.scipy.stats.gaussian_kde.logpdf "jax.scipy.stats.gaussian_kde.logpdf")(x) | Log probability density function |
| [`pdf`](jax.scipy.stats.gaussian_kde.pdf.html#jax.scipy.stats.gaussian_kde.pdf "jax.scipy.stats.gaussian_kde.pdf")(x) | Probability density function |
| [`resample`](jax.scipy.stats.gaussian_kde.resample.html#jax.scipy.stats.gaussian_kde.resample "jax.scipy.stats.gaussian_kde.resample")(key\[, shape\]) | Randomly sample a dataset from the estimated pdf |
| `set_bandwidth`(\[bw_method\]) | This method is not implemented in the JAX interface. |
| `tree_flatten`() |  |
| `tree_unflatten`(aux_data, children) |  |

Attributes

|              |     |
|--------------|-----|
| `neff`       |     |
| `dataset`    |     |
| `weights`    |     |
| `covariance` |     |
| `inv_cov`    |     |
| `d`          |     |
| `n`          |     |

[](jax.scipy.stats.uniform.ppf.html "previous page")

previous

jax.scipy.stats.uniform.ppf

[](jax.scipy.stats.gaussian_kde.evaluate.html "next page")

next

jax.scipy.stats.gaussian_kde.evaluate

Contents

- [`gaussian_kde`](#jax.scipy.stats.gaussian_kde)
  - [`gaussian_kde.__init__()`](#jax.scipy.stats.gaussian_kde.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
