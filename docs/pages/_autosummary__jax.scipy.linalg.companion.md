- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.linalg.companion

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.linalg.companion.rst "Download source file")
-  .pdf

# jax.scipy.linalg.companion

## Contents

- [`companion()`](#jax.scipy.linalg.companion)

# jax.scipy.linalg.companion[\#](#jax-scipy-linalg-companion "Link to this heading")

jax.scipy.linalg.companion(*a*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/linalg.py#L2662-L2697)[\#](#jax.scipy.linalg.companion "Link to this definition")  
Construct a companion matrix.

JAX implementation of [`scipy.linalg.companion()`](https://scipy.github.io/devdocs/reference/generated/scipy.linalg.companion.html#scipy.linalg.companion "(in SciPy v1.19.0.dev)").

Given polynomial coefficients \\a = \[a_0, a_1, \ldots, a\_{n-1}\]\\ with \\a_0 \neq 0\\, the companion matrix is the \\(n-1) \times (n-1)\\ matrix whose first row is \\-\[a_1, a_2, \ldots, a\_{n-1}\] / a_0\\ and whose first sub-diagonal is filled with ones.

Parameters:  
**a** (*ArrayLike*) – array of shape `(...,`` ``N)` with `N`` ``>=`` ``2` specifying the polynomial coefficients.

Returns:  
A companion matrix of shape `(...,`` ``N`` ``-`` ``1,`` ``N`` ``-`` ``1)`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Note

Unlike [`scipy.linalg.companion()`](https://scipy.github.io/devdocs/reference/generated/scipy.linalg.companion.html#scipy.linalg.companion "(in SciPy v1.19.0.dev)"), this function does not check at runtime that `a[...,`` ``0]` is non-zero; if the leading coefficient is zero, the result will contain `inf` or `nan` entries.

Examples

    >>> jax.scipy.linalg.companion(jnp.array([1., -10., 31., -30.]))
    Array([[ 10., -31.,  30.],
           [  1.,   0.,   0.],
           [  0.,   1.,   0.]], dtype=float32)

[](jax.scipy.linalg.circulant.html "previous page")

previous

jax.scipy.linalg.circulant

[](jax.scipy.linalg.convolution_matrix.html "next page")

next

jax.scipy.linalg.convolution_matrix

Contents

- [`companion()`](#jax.scipy.linalg.companion)

By The JAX authors

© Copyright 2024, The JAX Authors.\
