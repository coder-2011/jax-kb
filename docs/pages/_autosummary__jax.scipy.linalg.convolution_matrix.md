- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.linalg.convolution_matrix

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.linalg.convolution_matrix.rst "Download source file")
-  .pdf

# jax.scipy.linalg.convolution_matrix

## Contents

- [`convolution_matrix()`](#jax.scipy.linalg.convolution_matrix)

# jax.scipy.linalg.convolution_matrix[\#](#jax-scipy-linalg-convolution-matrix "Link to this heading")

jax.scipy.linalg.convolution_matrix(*a*, *n*, *mode='full'*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/linalg.py#L2814-L2881)[\#](#jax.scipy.linalg.convolution_matrix "Link to this definition")  
Construct a convolution matrix.

JAX implementation of [`scipy.linalg.convolution_matrix()`](https://scipy.github.io/devdocs/reference/generated/scipy.linalg.convolution_matrix.html#scipy.linalg.convolution_matrix "(in SciPy v1.19.0.dev)").

Builds a Toeplitz matrix \\A\\ such that `A`` ``@`` ``v` equals `jnp.convolve(a,`` ``v,`` ``mode)`. The returned array has `n` columns; the row count \\k\\ depends on `mode`:

- `'full'`: \\k = m + n - 1\\

- `'same'`: \\k = \max(m, n)\\

- `'valid'`: \\k = \max(m, n) - \min(m, n) + 1\\

where \\m\\ is the size of `a` along the last axis.

Parameters:  
- **a** (*ArrayLike*) – array of shape `(...,`` ``m)` to convolve. Must have `m`` ``>=`` ``1`.

- **n** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – number of columns in the output. Must be a positive integer.

- **mode** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – one of `'full'`, `'same'`, `'valid'`. Defaults to `'full'`.

Returns:  
A convolution matrix of shape `(...,`` ``k,`` ``n)`, where `k` depends on `mode` as described above.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

[`jax.scipy.linalg.toeplitz()`](jax.scipy.linalg.toeplitz.html#jax.scipy.linalg.toeplitz "jax.scipy.linalg.toeplitz")

Examples

    >>> jax.scipy.linalg.convolution_matrix(jnp.array([-1, 4, -2]), 5, mode='same')
    Array([[ 4, -1,  0,  0,  0],
           [-2,  4, -1,  0,  0],
           [ 0, -2,  4, -1,  0],
           [ 0,  0, -2,  4, -1],
           [ 0,  0,  0, -2,  4]], dtype=int32)

[](jax.scipy.linalg.companion.html "previous page")

previous

jax.scipy.linalg.companion

[](jax.scipy.linalg.det.html "next page")

next

jax.scipy.linalg.det

Contents

- [`convolution_matrix()`](#jax.scipy.linalg.convolution_matrix)

By The JAX authors

© Copyright 2024, The JAX Authors.\
