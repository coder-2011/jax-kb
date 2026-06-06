- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.linalg.dft

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.linalg.dft.rst "Download source file")
-  .pdf

# jax.scipy.linalg.dft

## Contents

- [`dft()`](#jax.scipy.linalg.dft)

# jax.scipy.linalg.dft[\#](#jax-scipy-linalg-dft "Link to this heading")

jax.scipy.linalg.dft(*n*, *scale=None*, *\**, *dtype=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/linalg.py#L3171-L3217)[\#](#jax.scipy.linalg.dft "Link to this definition")  
Construct an n-by-n discrete Fourier transform matrix.

JAX implementation of [`scipy.linalg.dft()`](https://scipy.github.io/devdocs/reference/generated/scipy.linalg.dft.html#scipy.linalg.dft "(in SciPy v1.19.0.dev)").

The DFT matrix \\W_n\\ has entries \\W\_{ij} = \omega^{ij}\\, where \\\omega = e^{-2\pi i / n}\\ is the primitive n-th root of unity, for \\0 \le i, j \< n\\.

Parameters:  
- **n** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – size of the matrix.

- **scale** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – (optional) `None` (default, unscaled), `'sqrtn'` (scale by \\1/\sqrt{n}\\, making the matrix unitary), or `'n'` (scale by \\1/n\\).

- **dtype** (*DTypeLike* *\|* *None*) – (optional) complex floating-point dtype for the output. Defaults to JAX’s default complex dtype.

Returns:  
A DFT matrix of shape `(n,`` ``n)`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Examples

    >>> jax.scipy.linalg.dft(4).round(3)
    Array([[ 1.+0.j,  1.+0.j,  1.+0.j,  1.+0.j],
           [ 1.+0.j, -0.-1.j, -1.+0.j,  0.+1.j],
           [ 1.+0.j, -1.+0.j,  1.-0.j, -1.+0.j],
           [ 1.+0.j,  0.+1.j, -1.+0.j, -0.-1.j]], dtype=complex64)

[](jax.scipy.linalg.det.html "previous page")

previous

jax.scipy.linalg.det

[](jax.scipy.linalg.eigh.html "next page")

next

jax.scipy.linalg.eigh

Contents

- [`dft()`](#jax.scipy.linalg.dft)

By The JAX authors

© Copyright 2024, The JAX Authors.\
