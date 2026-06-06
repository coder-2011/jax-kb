- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.linalg.leslie

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.linalg.leslie.rst "Download source file")
-  .pdf

# jax.scipy.linalg.leslie

## Contents

- [`leslie()`](#jax.scipy.linalg.leslie)

# jax.scipy.linalg.leslie[\#](#jax-scipy-linalg-leslie "Link to this heading")

jax.scipy.linalg.leslie(*f*, *s*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/linalg.py#L2617-L2655)[\#](#jax.scipy.linalg.leslie "Link to this definition")  
Construct a Leslie matrix.

JAX implementation of [`scipy.linalg.leslie()`](https://scipy.github.io/devdocs/reference/generated/scipy.linalg.leslie.html#scipy.linalg.leslie "(in SciPy v1.19.0.dev)").

Given fecundity coefficients `f` of shape `(...,`` ``N)` and survival coefficients `s` of shape `(...,`` ``N`` ``-`` ``1)`, the Leslie matrix has `f` as its first row, `s` along its first sub-diagonal, and zeros elsewhere.

Parameters:  
- **f** (*ArrayLike*) – array of shape `(...,`` ``N)` with `N`` ``>=`` ``2` containing the fecundity coefficients.

- **s** (*ArrayLike*) – array of shape `(...,`` ``N`` ``-`` ``1)` containing the survival coefficients.

Returns:  
A Leslie matrix of shape `(...,`` ``N,`` ``N)`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Examples

    >>> jax.scipy.linalg.leslie(jnp.array([0.1, 2.0, 1.0, 0.1]),
    ...                         jnp.array([0.2, 0.8, 0.7]))
    Array([[0.1, 2. , 1. , 0.1],
           [0.2, 0. , 0. , 0. ],
           [0. , 0.8, 0. , 0. ],
           [0. , 0. , 0.7, 0. ]], dtype=float32)

[](jax.scipy.linalg.invpascal.html "previous page")

previous

jax.scipy.linalg.invpascal

[](jax.scipy.linalg.lu.html "next page")

next

jax.scipy.linalg.lu

Contents

- [`leslie()`](#jax.scipy.linalg.leslie)

By The JAX authors

© Copyright 2024, The JAX Authors.\
