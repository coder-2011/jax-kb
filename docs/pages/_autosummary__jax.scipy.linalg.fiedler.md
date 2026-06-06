- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.linalg.fiedler

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.linalg.fiedler.rst "Download source file")
-  .pdf

# jax.scipy.linalg.fiedler

## Contents

- [`fiedler()`](#jax.scipy.linalg.fiedler)

# jax.scipy.linalg.fiedler[\#](#jax-scipy-linalg-fiedler "Link to this heading")

jax.scipy.linalg.fiedler(*a*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/linalg.py#L2706-L2732)[\#](#jax.scipy.linalg.fiedler "Link to this definition")  
Construct a symmetric Fiedler matrix.

JAX implementation of [`scipy.linalg.fiedler()`](https://scipy.github.io/devdocs/reference/generated/scipy.linalg.fiedler.html#scipy.linalg.fiedler "(in SciPy v1.19.0.dev)").

The Fiedler matrix has entries \\F\_{ij} = \|a_i - a_j\|\\ for \\0 \le i, j \< n\\, where `a` is the input vector. The result is symmetric with a zero diagonal.

Parameters:  
**a** (*ArrayLike*) – array of shape `(...,`` ``N)`.

Returns:  
A Fiedler matrix of shape `(...,`` ``N,`` ``N)`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Examples

    >>> jax.scipy.linalg.fiedler(jnp.array([1, 4, 12, 45, 77]))
    Array([[ 0,  3, 11, 44, 76],
           [ 3,  0,  8, 41, 73],
           [11,  8,  0, 33, 65],
           [44, 41, 33,  0, 32],
           [76, 73, 65, 32,  0]], dtype=int32)

[](jax.scipy.linalg.expm_frechet.html "previous page")

previous

jax.scipy.linalg.expm_frechet

[](jax.scipy.linalg.fiedler_companion.html "next page")

next

jax.scipy.linalg.fiedler_companion

Contents

- [`fiedler()`](#jax.scipy.linalg.fiedler)

By The JAX authors

© Copyright 2024, The JAX Authors.\
