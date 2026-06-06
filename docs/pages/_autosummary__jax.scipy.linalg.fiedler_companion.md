- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.linalg.fiedler_companion

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.linalg.fiedler_companion.rst "Download source file")
-  .pdf

# jax.scipy.linalg.fiedler_companion

## Contents

- [`fiedler_companion()`](#jax.scipy.linalg.fiedler_companion)

# jax.scipy.linalg.fiedler_companion[\#](#jax-scipy-linalg-fiedler-companion "Link to this heading")

jax.scipy.linalg.fiedler_companion(*a*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/linalg.py#L2734-L2775)[\#](#jax.scipy.linalg.fiedler_companion "Link to this definition")  
Construct a Fiedler companion matrix.

JAX implementation of [`scipy.linalg.fiedler_companion()`](https://scipy.github.io/devdocs/reference/generated/scipy.linalg.fiedler_companion.html#scipy.linalg.fiedler_companion "(in SciPy v1.19.0.dev)").

Given polynomial coefficients \\a = \[a_0, a_1, \ldots, a\_{n}\]\\ with \\a_0 \neq 0\\, this constructs a pentadiagonal matrix whose eigenvalues coincide with the roots of the polynomial. The result is similar to [`companion()`](jax.scipy.linalg.companion.html#jax.scipy.linalg.companion "jax.scipy.linalg.companion") but with a sparser, banded structure.

Parameters:  
**a** (*ArrayLike*) – array of shape `(...,`` ``N)` specifying the polynomial coefficients in descending order. The last axis must have nonzero length. For `N`` ``==`` ``1` an empty `(0,`` ``0)` matrix is returned along that slice.

Raises:  
[**ValueError**](https://docs.python.org/3/library/exceptions.html#ValueError "(in Python v3.14)") – if the last axis of `a` has length zero.

Returns:  
A Fiedler companion matrix of shape `(...,`` ``N`` ``-`` ``1,`` ``N`` ``-`` ``1)`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Note

Unlike [`scipy.linalg.fiedler_companion()`](https://scipy.github.io/devdocs/reference/generated/scipy.linalg.fiedler_companion.html#scipy.linalg.fiedler_companion "(in SciPy v1.19.0.dev)"), this function does not check at runtime that `a[...,`` ``0]` is non-zero; if the leading coefficient is zero, the result will contain `inf` or `nan` entries.

Examples

    >>> a = jnp.array([1., -16., 86., -176., 105.])
    >>> jax.scipy.linalg.fiedler_companion(a)
    Array([[ 16., -86.,   1.,   0.],
           [  1.,   0.,   0.,   0.],
           [  0., 176.,   0., -105.],
           [  0.,   1.,   0.,   0.]], dtype=float32)

[](jax.scipy.linalg.fiedler.html "previous page")

previous

jax.scipy.linalg.fiedler

[](jax.scipy.linalg.funm.html "next page")

next

jax.scipy.linalg.funm

Contents

- [`fiedler_companion()`](#jax.scipy.linalg.fiedler_companion)

By The JAX authors

© Copyright 2024, The JAX Authors.\
