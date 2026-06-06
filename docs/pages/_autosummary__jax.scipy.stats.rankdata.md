- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.stats.rankdata

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.stats.rankdata.rst "Download source file")
-  .pdf

# jax.scipy.stats.rankdata

## Contents

- [`rankdata()`](#jax.scipy.stats.rankdata)

# jax.scipy.stats.rankdata[\#](#jax-scipy-stats-rankdata "Link to this heading")

jax.scipy.stats.rankdata(*a*, *method='average'*, *\**, *axis=None*, *nan_policy='propagate'*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/stats/_core.py#L143-L227)[\#](#jax.scipy.stats.rankdata "Link to this definition")  
Compute the rank of data along an array axis.

JAX implementation of [`scipy.stats.rankdata()`](https://scipy.github.io/devdocs/reference/generated/scipy.stats.rankdata.html#scipy.stats.rankdata "(in SciPy v1.19.0.dev)").

Ranks begin at 1, and the *method* argument controls how ties are handled.

Parameters:  
- **a** (*ArrayLike*) – arraylike

- **method** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – str, default=”average”. Supported methods are `("average",`` ``"min",`` ``"max",`` ``"dense",`` ``"ordinal")` For details, see the [`scipy.stats.rankdata()`](https://scipy.github.io/devdocs/reference/generated/scipy.stats.rankdata.html#scipy.stats.rankdata "(in SciPy v1.19.0.dev)") documentation.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – optional integer. If not specified, the input array is flattened.

- **nan_policy** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – str, JAX’s implementation only supports `"propagate"`.

Returns:  
array of ranks along the specified axis.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Examples

    >>> x = jnp.array([10, 30, 20])
    >>> rankdata(x)
    Array([1., 3., 2.], dtype=float32)

    >>> x = jnp.array([1, 3, 2, 3])
    >>> rankdata(x)
    Array([1. , 3.5, 2. , 3.5], dtype=float32)

[](jax.scipy.stats.mode.html "previous page")

previous

jax.scipy.stats.mode

[](jax.scipy.stats.sem.html "next page")

next

jax.scipy.stats.sem

Contents

- [`rankdata()`](#jax.scipy.stats.rankdata)

By The JAX authors

© Copyright 2024, The JAX Authors.\
