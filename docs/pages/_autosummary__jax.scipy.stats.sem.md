- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.stats.sem

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.stats.sem.rst "Download source file")
-  .pdf

# jax.scipy.stats.sem

## Contents

- [`sem()`](#jax.scipy.stats.sem)

# jax.scipy.stats.sem[\#](#jax-scipy-stats-sem "Link to this heading")

jax.scipy.stats.sem(*a*, *axis=0*, *ddof=1*, *nan_policy='propagate'*, *\**, *keepdims=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/stats/_core.py#L229-L312)[\#](#jax.scipy.stats.sem "Link to this definition")  
Compute the standard error of the mean.

JAX implementation of [`scipy.stats.sem()`](https://scipy.github.io/devdocs/reference/generated/scipy.stats.sem.html#scipy.stats.sem "(in SciPy v1.19.0.dev)").

Parameters:  
- **a** (*ArrayLike*) – arraylike

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – optional integer. If not specified, the input array is flattened.

- **ddof** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – integer, default=1. The degrees of freedom in the SEM computation.

- **nan_policy** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – str, default=”propagate”. JAX supports only “propagate” and “omit”.

- **keepdims** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – bool, default=False. If true, reduced axes are left in the result with size 1.

Returns:  
array

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Examples

    >>> x = jnp.array([2, 4, 1, 1, 3, 4, 4, 2, 3])
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   jax.scipy.stats.sem(x)
    Array(0.41, dtype=float32)

For multi dimensional arrays, `sem` computes standard error of mean along `axis=0`:

    >>> x1 = jnp.array([[1, 2, 1, 3, 2, 1],
    ...                 [3, 1, 3, 2, 1, 3],
    ...                 [1, 2, 2, 3, 1, 2]])
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   jax.scipy.stats.sem(x1)
    Array([0.67, 0.33, 0.58, 0.33, 0.33, 0.58], dtype=float32)

If `axis=1`, standard error of mean will be computed along `axis`` ``1`.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   jax.scipy.stats.sem(x1, axis=1)
    Array([0.33, 0.4 , 0.31], dtype=float32)

If `axis=None`, standard error of mean will be computed along all the axes.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   jax.scipy.stats.sem(x1, axis=None)
    Array(0.2, dtype=float32)

By default, `sem` reduces the dimension of the result. To keep the dimensions same as that of the input array, the argument `keepdims` must be set to `True`.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   jax.scipy.stats.sem(x1, axis=1, keepdims=True)
    Array([[0.33],
           [0.4 ],
           [0.31]], dtype=float32)

Since, by default, `nan_policy='propagate'`, `sem` propagates the `nan` values in the result.

    >>> nan = np.nan
    >>> x2 = jnp.array([[1, 2, 3, nan, 4, 2],
    ...                 [4, 5, 4, 3, nan, 1],
    ...                 [7, nan, 8, 7, 9, nan]])
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   jax.scipy.stats.sem(x2)
    Array([1.73,  nan, 1.53,  nan,  nan,  nan], dtype=float32)

If `` nan_policy='omit` ``, `sem` omits the `nan` values and computes the error for the remaining values along the specified axis.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   jax.scipy.stats.sem(x2, nan_policy='omit')
    Array([1.73, 1.5 , 1.53, 2.  , 2.5 , 0.5 ], dtype=float32)

[](jax.scipy.stats.rankdata.html "previous page")

previous

jax.scipy.stats.rankdata

[](jax.scipy.stats.bernoulli.logpmf.html "next page")

next

jax.scipy.stats.bernoulli.logpmf

Contents

- [`sem()`](#jax.scipy.stats.sem)

By The JAX authors

© Copyright 2024, The JAX Authors.\
