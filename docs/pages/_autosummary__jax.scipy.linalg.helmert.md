- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.linalg.helmert

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.linalg.helmert.rst "Download source file")
-  .pdf

# jax.scipy.linalg.helmert

## Contents

- [`helmert()`](#jax.scipy.linalg.helmert)

# jax.scipy.linalg.helmert[\#](#jax-scipy-linalg-helmert "Link to this heading")

jax.scipy.linalg.helmert(*n*, *full=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/linalg.py#L3096-L3134)[\#](#jax.scipy.linalg.helmert "Link to this definition")  
Construct a Helmert matrix.

JAX implementation of [`scipy.linalg.helmert()`](https://scipy.github.io/devdocs/reference/generated/scipy.linalg.helmert.html#scipy.linalg.helmert "(in SciPy v1.19.0.dev)").

The Helmert matrix has rows of orthonormal contrasts. For `k`` ``=`` ``1,`` ``...,`` ``n`` ``-`` ``1`, row `k`` ``-`` ``1` is \\(\underbrace{1, \ldots, 1}\_{k}, -k, 0, \ldots, 0) / \sqrt{k(k + 1)}\\. If `full` is `True`, a row of \\1 / \sqrt{n}\\ is prepended.

Parameters:  
- **n** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – size of the matrix. Must be a positive integer.

- **full** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – if `True`, return the full `(n,`` ``n)` matrix; otherwise return only the `(n`` ``-`` ``1,`` ``n)` contrast block. Defaults to `False`.

Returns:  
A Helmert matrix of shape `(n`` ``-`` ``1,`` ``n)` if `full` is `False`, or `(n,`` ``n)` if `full` is `True`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Examples

    >>> jax.scipy.linalg.helmert(2, full=True).round(3)
    Array([[ 0.707,  0.707],
           [ 0.707, -0.707]], dtype=float32)

[](jax.scipy.linalg.hadamard.html "previous page")

previous

jax.scipy.linalg.hadamard

[](jax.scipy.linalg.hessenberg.html "next page")

next

jax.scipy.linalg.hessenberg

Contents

- [`helmert()`](#jax.scipy.linalg.helmert)

By The JAX authors

© Copyright 2024, The JAX Authors.\
