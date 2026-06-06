- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.stats.mode

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.stats.mode.rst "Download source file")
-  .pdf

# jax.scipy.stats.mode

## Contents

- [`mode()`](#jax.scipy.stats.mode)

# jax.scipy.stats.mode[\#](#jax-scipy-stats-mode "Link to this heading")

jax.scipy.stats.mode(*a*, *axis=0*, *nan_policy='propagate'*, *keepdims=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/stats/_core.py#L36-L137)[\#](#jax.scipy.stats.mode "Link to this definition")  
Compute the mode (most common value) along an axis of an array.

JAX implementation of [`scipy.stats.mode()`](https://scipy.github.io/devdocs/reference/generated/scipy.stats.mode.html#scipy.stats.mode "(in SciPy v1.19.0.dev)").

Parameters:  
- **a** (*ArrayLike*) – arraylike

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – int, default=0. Axis along which to compute the mode.

- **nan_policy** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – str. JAX only supports `"propagate"`.

- **keepdims** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – bool, default=False. If true, reduced axes are left in the result with size 1.

Returns:  
A tuple of arrays, `(mode,`` ``count)`. `mode` is the array of modal values, and `count` is the number of times each value appears in the input array.

Return type:  
ModeResult

Examples

    >>> x = jnp.array([2, 4, 1, 1, 3, 4, 4, 2, 3])
    >>> mode, count = jax.scipy.stats.mode(x)
    >>> mode, count
    (Array(4, dtype=int32), Array(3, dtype=int32))

For multi dimensional arrays, `jax.scipy.stats.mode` computes the `mode` and the corresponding `count` along `axis=0`:

    >>> x1 = jnp.array([[1, 2, 1, 3, 2, 1],
    ...                 [3, 1, 3, 2, 1, 3],
    ...                 [1, 2, 2, 3, 1, 2]])
    >>> mode, count = jax.scipy.stats.mode(x1)
    >>> mode, count
    (Array([1, 2, 1, 3, 1, 1], dtype=int32), Array([2, 2, 1, 2, 2, 1], dtype=int32))

If `axis=1`, `mode` and `count` will be computed along `axis`` ``1`.

    >>> mode, count = jax.scipy.stats.mode(x1, axis=1)
    >>> mode, count
    (Array([1, 3, 2], dtype=int32), Array([3, 3, 3], dtype=int32))

By default, `jax.scipy.stats.mode` reduces the dimension of the result. To keep the dimensions same as that of the input array, the argument `keepdims` must be set to `True`.

    >>> mode, count = jax.scipy.stats.mode(x1, axis=1, keepdims=True)
    >>> mode, count
    (Array([[1],
           [3],
           [2]], dtype=int32), Array([[3],
           [3],
           [3]], dtype=int32))

[](jax.scipy.special.zeta.html "previous page")

previous

jax.scipy.special.zeta

[](jax.scipy.stats.rankdata.html "next page")

next

jax.scipy.stats.rankdata

Contents

- [`mode()`](#jax.scipy.stats.mode)

By The JAX authors

© Copyright 2024, The JAX Authors.\
