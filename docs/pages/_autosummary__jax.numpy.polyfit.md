- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.polyfit

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.polyfit.rst "Download source file")
-  .pdf

# jax.numpy.polyfit

## Contents

- [`polyfit()`](#jax.numpy.polyfit)

# jax.numpy.polyfit[\#](#jax-numpy-polyfit "Link to this heading")

jax.numpy.polyfit(*x*, *y*, *deg*, *rcond=None*, *full=False*, *w=None*, *cov=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/polynomial.py#L126-L305)[\#](#jax.numpy.polyfit "Link to this definition")  
Least squares polynomial fit to data.

Jax implementation of [`numpy.polyfit()`](https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html#numpy.polyfit "(in NumPy v2.4)").

Given a set of data points `(x,`` ``y)` and degree of polynomial `deg`, the function finds a polynomial equation of the form:

\\y = p(x) = p\[0\] x^{deg} + p\[1\] x^{deg - 1} + ... + p\[deg\]\\

Parameters:  
- **x** (*ArrayLike*) – Array of data points of shape `(M,)`.

- **y** (*ArrayLike*) – Array of data points of shape `(M,)` or `(M,`` ``K)`.

- **deg** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – Degree of the polynomials. It must be specified statically.

- **rcond** ([*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* *None*) – Relative condition number of the fit. Default value is `len(x)`` ``*`` ``eps`. It must be specified statically.

- **full** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – Switch that controls the return value. Default is `False` which restricts the return value to the array of polynomial coefficients `p`. If `True`, the function returns a tuple `(p,`` ``resids,`` ``rank,`` ``s,`` ``rcond)`. It must be specified statically.

- **w** (*ArrayLike* *\|* *None*) – Array of weights of shape `(M,)`. If None, all data points are considered to have equal weight. If not None, the weight \\w_i\\ is applied to the unsquared residual of \\y_i - \widehat{y}\_i\\ at \\x_i\\, where \\\widehat{y}\_i\\ is the fitted value of \\y_i\\. Default is None.

- **cov** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – Boolean or string. If `True`, returns the covariance matrix scaled by `resids/(M-deg-1)` along with polynomial coefficients. If `cov='unscaled'`, returns the unscaled version of covariance matrix. Default is `False`. `cov` is ignored if `full=True`. It must be specified statically.

Returns:  
- An array polynomial coefficients `p` if `full=False` and `cov=False`.

- A tuple of arrays `(p,`` ``resids,`` ``rank,`` ``s,`` ``rcond)` if `full=True`. Where

  - `p` is an array of shape `(M,)` or `(M,`` ``K)` containing the polynomial coefficients.

  - `resids` is the sum of squared residual of shape () or (K,).

  - `rank` is the rank of the matrix `x`.

  - `s` is the singular values of the matrix `x`.

  - `rcond` as the array.

- A tuple of arrays `(p,`` ``C)` if `full=False` and `cov=True`. Where

  - `p` is an array of shape `(M,)` or `(M,`` ``K)` containing the polynomial coefficients.

  - `C` is the covariance matrix of polynomial coefficients of shape `(deg`` ``+`` ``1,`` ``deg`` ``+`` ``1)` or `(deg`` ``+`` ``1,`` ``deg`` ``+`` ``1,`` ``1)`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array") \| [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), …\]

Note

Unlike [`numpy.polyfit()`](https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html#numpy.polyfit "(in NumPy v2.4)") implementation of polyfit, [`jax.numpy.polyfit()`](#jax.numpy.polyfit "jax.numpy.polyfit") will not warn on rank reduction, which indicates an ill conditioned matrix.

See also

- [`jax.numpy.poly()`](jax.numpy.poly.html#jax.numpy.poly "jax.numpy.poly"): Finds the polynomial coefficients of the given sequence of roots.

- [`jax.numpy.polyval()`](jax.numpy.polyval.html#jax.numpy.polyval "jax.numpy.polyval"): Evaluate a polynomial at specific values.

- [`jax.numpy.roots()`](jax.numpy.roots.html#jax.numpy.roots "jax.numpy.roots"): Computes the roots of a polynomial for given coefficients.

Examples

    >>> x = jnp.array([3., 6., 9., 4.])
    >>> y = jnp.array([[0, 1, 2],
    ...                [2, 5, 7],
    ...                [8, 4, 9],
    ...                [1, 6, 3]])
    >>> p = jnp.polyfit(x, y, 2)
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(p)
    [[ 0.2  -0.35 -0.14]
     [-1.17  4.47  2.96]
     [ 1.95 -8.21 -5.93]]

If `full=True`, returns a tuple of arrays as follows:

    >>> p, resids, rank, s, rcond = jnp.polyfit(x, y, 2, full=True)
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print("Polynomial Coefficients:", "\n", p, "\n",
    ...         "Residuals:", resids, "\n",
    ...         "Rank:", rank, "\n",
    ...         "s:", s, "\n",
    ...         "rcond:", rcond)
    Polynomial Coefficients:
    [[ 0.2  -0.35 -0.14]
    [-1.17  4.47  2.96]
    [ 1.95 -8.21 -5.93]]
    Residuals: [0.37 5.94 0.61]
    Rank: 3
    s: [1.67 0.47 0.04]
    rcond: 4.7683716e-07

If `cov=True` and `full=False`, returns a tuple of arrays having polynomial coefficients and covariance matrix.

    >>> p, C = jnp.polyfit(x, y, 2, cov=True)
    >>> p.shape, C.shape
    ((3, 3), (3, 3, 3))

[](jax.numpy.polydiv.html "previous page")

previous

jax.numpy.polydiv

[](jax.numpy.polyint.html "next page")

next

jax.numpy.polyint

Contents

- [`polyfit()`](#jax.numpy.polyfit)

By The JAX authors

© Copyright 2024, The JAX Authors.\
