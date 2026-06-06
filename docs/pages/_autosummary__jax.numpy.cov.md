- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.cov

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.cov.rst "Download source file")
-  .pdf

# jax.numpy.cov

## Contents

- [`cov()`](#jax.numpy.cov)

# jax.numpy.cov[\#](#jax-numpy-cov "Link to this heading")

jax.numpy.cov(*m*, *y=None*, *rowvar=True*, *bias=False*, *ddof=None*, *fweights=None*, *aweights=None*, *dtype=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L9054-L9225)[\#](#jax.numpy.cov "Link to this definition")  
Estimate the weighted sample covariance.

JAX implementation of [`numpy.cov()`](https://numpy.org/doc/stable/reference/generated/numpy.cov.html#numpy.cov "(in NumPy v2.4)").

The covariance \\C\_{ij}\\ between variable *i* and variable *j* is defined as

\\cov\[X_i, X_j\] = E\[(X_i - E\[X_i\])(X_j - E\[X_j\])\]\\

Given an array of *N* observations of the variables \\X_i\\ and \\X_j\\, this can be estimated via the sample covariance:

\\C\_{ij} = \frac{1}{N - 1} \sum\_{n=1}^N (X\_{in} - \overline{X_i})(X\_{jn} - \overline{X_j})\\

Where \\\overline{X_i} = \frac{1}{N} \sum\_{k=1}^N X\_{ik}\\ is the mean of the observations.

Parameters:  
- **m** (*ArrayLike*) – array of shape `(M,`` ``N)` (if `rowvar` is True), or `(N,`` ``M)` (if `rowvar` is False) representing `N` observations of `M` variables. `m` may also be one-dimensional, representing `N` observations of a single variable.

- **y** (*ArrayLike* *\|* *None*) – optional set of additional observations, with the same form as `m`. If specified, then `y` is combined with `m`, i.e. for the default `rowvar`` ``=`` ``True` case, `m` becomes `jnp.vstack([m,`` ``y])`.

- **rowvar** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – if True (default) then each row of `m` represents a variable. If False, then each column represents a variable.

- **bias** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – if False (default) then normalize the covariance by `N`` ``-`` ``1`. If True, then normalize the covariance by `N`

- **ddof** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – specify the degrees of freedom. Defaults to `1` if `bias` is False, or to `0` if `bias` is True.

- **fweights** (*ArrayLike* *\|* *None*) – optional array of integer frequency weights of shape `(N,)`. This is an absolute weight specifying the number of times each observation is included in the computation.

- **aweights** (*ArrayLike* *\|* *None*) – optional array of observation weights of shape `(N,)`. This is a relative weight specifying the “importance” of each observation. In the `ddof=0` case, it is equivalent to assigning probabilities to each observation.

- **dtype** (*DTypeLike* *\|* *None*) – optional data type of the result. Must be a float or complex type; if not specified, it will be determined based on the dtype of the input.

Returns:  
A covariance matrix of shape `(M,`` ``M)`, or a scalar with shape `()` if `M`` ``=`` ``1`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.corrcoef()`](jax.numpy.corrcoef.html#jax.numpy.corrcoef "jax.numpy.corrcoef"): compute the correlation coefficient, a normalized version of the covariance matrix.

Examples

Consider these observations of two variables that correlate perfectly. The covariance matrix in this case is a 2x2 matrix of ones:

    >>> x = jnp.array([[0, 1, 2],
    ...                [0, 1, 2]])
    >>> jnp.cov(x)
    Array([[1., 1.],
           [1., 1.]], dtype=float32)

Now consider these observations of two variables that are perfectly anti-correlated. The covariance matrix in this case has `-1` in the off-diagonal:

    >>> x = jnp.array([[-1,  0,  1],
    ...                [ 1,  0, -1]])
    >>> jnp.cov(x)
    Array([[ 1., -1.],
           [-1.,  1.]], dtype=float32)

Equivalently, these sequences can be specified as separate arguments, in which case they are stacked before continuing the computation.

    >>> x = jnp.array([-1, 0, 1])
    >>> y = jnp.array([1, 0, -1])
    >>> jnp.cov(x, y)
    Array([[ 1., -1.],
           [-1.,  1.]], dtype=float32)

In general, the entries of the covariance matrix may be any positive or negative real value. For example, here is the covariance of 100 points drawn from a 3-dimensional standard normal distribution:

    >>> key = jax.random.key(0)
    >>> x = jax.random.normal(key, shape=(3, 100))
    >>> with jnp.printoptions(precision=2):
    ...   print(jnp.cov(x))
    [[0.9  0.03 0.1 ]
     [0.03 1.   0.01]
     [0.1  0.01 0.85]]

[](jax.numpy.count_nonzero.html "previous page")

previous

jax.numpy.count_nonzero

[](jax.numpy.cross.html "next page")

next

jax.numpy.cross

Contents

- [`cov()`](#jax.numpy.cov)

By The JAX authors

© Copyright 2024, The JAX Authors.\
