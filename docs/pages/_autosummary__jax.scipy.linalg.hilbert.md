- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.linalg.hilbert

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.linalg.hilbert.rst "Download source file")
-  .pdf

# jax.scipy.linalg.hilbert

## Contents

- [`hilbert()`](#jax.scipy.linalg.hilbert)

# jax.scipy.linalg.hilbert[\#](#jax-scipy-linalg-hilbert "Link to this heading")

jax.scipy.linalg.hilbert(*n*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/linalg.py#L2883-L2914)[\#](#jax.scipy.linalg.hilbert "Link to this definition")  
Create a Hilbert matrix of order n.

JAX implementation of [`scipy.linalg.hilbert()`](https://scipy.github.io/devdocs/reference/generated/scipy.linalg.hilbert.html#scipy.linalg.hilbert "(in SciPy v1.19.0.dev)").

The Hilbert matrix is defined by:

\\H\_{ij} = \frac{1}{i + j + 1}\\

for \\1 \le i \le n\\ and \\1 \le j \le n\\.

Parameters:  
**n** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – the size of the matrix to create.

Returns:  
A Hilbert matrix of shape `(n,`` ``n)`

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Examples

    >>> jax.scipy.linalg.hilbert(2)
    Array([[1.        , 0.5       ],
           [0.5       , 0.33333334]], dtype=float32)
    >>> jax.scipy.linalg.hilbert(3)
    Array([[1.        , 0.5       , 0.33333334],
           [0.5       , 0.33333334, 0.25      ],
           [0.33333334, 0.25      , 0.2       ]], dtype=float32)

[](jax.scipy.linalg.hankel.html "previous page")

previous

jax.scipy.linalg.hankel

[](jax.scipy.linalg.inv.html "next page")

next

jax.scipy.linalg.inv

Contents

- [`hilbert()`](#jax.scipy.linalg.hilbert)

By The JAX authors

© Copyright 2024, The JAX Authors.\
