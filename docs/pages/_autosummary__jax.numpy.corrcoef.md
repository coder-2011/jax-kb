- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.corrcoef

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.corrcoef.rst "Download source file")
-  .pdf

# jax.numpy.corrcoef

## Contents

- [`corrcoef()`](#jax.numpy.corrcoef)

# jax.numpy.corrcoef[\#](#jax-numpy-corrcoef "Link to this heading")

jax.numpy.corrcoef(*x*, *y=None*, *rowvar=True*, *dtype=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L9227-L9324)[\#](#jax.numpy.corrcoef "Link to this definition")  
Compute the Pearson correlation coefficients.

JAX implementation of [`numpy.corrcoef()`](https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html#numpy.corrcoef "(in NumPy v2.4)").

This is a normalized version of the sample covariance computed by [`jax.numpy.cov()`](jax.numpy.cov.html#jax.numpy.cov "jax.numpy.cov"). For a sample covariance \\C\_{ij}\\, the correlation coefficients are

\\R\_{ij} = \frac{C\_{ij}}{\sqrt{C\_{ii}C\_{jj}}}\\

they are constructed such that the values satisfy \\-1 \le R\_{ij} \le 1\\.

Parameters:  
- **x** (*ArrayLike*) – array of shape `(M,`` ``N)` (if `rowvar` is True), or `(N,`` ``M)` (if `rowvar` is False) representing `N` observations of `M` variables. `x` may also be one-dimensional, representing `N` observations of a single variable.

- **y** (*ArrayLike* *\|* *None*) – optional set of additional observations, with the same form as `m`. If specified, then `y` is combined with `m`, i.e. for the default `rowvar`` ``=`` ``True` case, `m` becomes `jnp.vstack([m,`` ``y])`.

- **rowvar** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – if True (default) then each row of `m` represents a variable. If False, then each column represents a variable.

- **dtype** (*DTypeLike* *\|* *None*) – optional data type of the result. Must be a float or complex type; if not specified, it will be determined based on the dtype of the input.

Returns:  
A covariance matrix of shape `(M,`` ``M)`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.cov()`](jax.numpy.cov.html#jax.numpy.cov "jax.numpy.cov"): compute the covariance matrix.

Examples

Consider these observations of two variables that correlate perfectly. The correlation matrix in this case is a 2x2 matrix of ones:

    >>> x = jnp.array([[0, 1, 2],
    ...                [0, 1, 2]])
    >>> jnp.corrcoef(x)
    Array([[1., 1.],
           [1., 1.]], dtype=float32)

Now consider these observations of two variables that are perfectly anti-correlated. The correlation matrix in this case has `-1` in the off-diagonal:

    >>> x = jnp.array([[-1,  0,  1],
    ...                [ 1,  0, -1]])
    >>> jnp.corrcoef(x)
    Array([[ 1., -1.],
           [-1.,  1.]], dtype=float32)

Equivalently, these sequences can be specified as separate arguments, in which case they are stacked before continuing the computation.

    >>> x = jnp.array([-1, 0, 1])
    >>> y = jnp.array([1, 0, -1])
    >>> jnp.corrcoef(x, y)
    Array([[ 1., -1.],
           [-1.,  1.]], dtype=float32)

The entries of the correlation matrix are normalized such that they lie within the range -1 to +1, where +1 indicates perfect correlation and -1 indicates perfect anti-correlation. For example, here is the correlation of 100 points drawn from a 3-dimensional standard normal distribution:

    >>> key = jax.random.key(0)
    >>> x = jax.random.normal(key, shape=(3, 100))
    >>> with jnp.printoptions(precision=2):
    ...   print(jnp.corrcoef(x))
    [[1.   0.03 0.12]
     [0.03 1.   0.01]
     [0.12 0.01 1.  ]]

[](jax.numpy.copysign.html "previous page")

previous

jax.numpy.copysign

[](jax.numpy.correlate.html "next page")

next

jax.numpy.correlate

Contents

- [`corrcoef()`](#jax.numpy.corrcoef)

By The JAX authors

© Copyright 2024, The JAX Authors.\
