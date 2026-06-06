- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.linalg.det

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.linalg.det.rst "Download source file")
-  .pdf

# jax.scipy.linalg.det

## Contents

- [`det()`](#jax.scipy.linalg.det)

# jax.scipy.linalg.det[\#](#jax-scipy-linalg-det "Link to this heading")

jax.scipy.linalg.det(*a*, *overwrite_a=False*, *check_finite=True*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/linalg.py#L346-L381)[\#](#jax.scipy.linalg.det "Link to this definition")  
Compute the determinant of a matrix

JAX implementation of [`scipy.linalg.det()`](https://scipy.github.io/devdocs/reference/generated/scipy.linalg.det.html#scipy.linalg.det "(in SciPy v1.19.0.dev)").

Parameters:  
- **a** (*ArrayLike*) – input array, of shape `(...,`` ``N,`` ``N)`

- **overwrite_a** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – unused by JAX

- **check_finite** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – unused by JAX

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Returns  
Determinant of shape `a.shape[:-2]`

See also

[`jax.numpy.linalg.det()`](jax.numpy.linalg.det.html#jax.numpy.linalg.det "jax.numpy.linalg.det"): NumPy-style determinant API

Examples

Determinant of a small 2D array:

    >>> x = jnp.array([[1., 2.],
    ...                [3., 4.]])
    >>> jax.scipy.linalg.det(x)
    Array(-2., dtype=float32)

Batch-wise determinant of multiple 2D arrays:

    >>> x = jnp.array([[[1., 2.],
    ...                 [3., 4.]],
    ...                [[8., 5.],
    ...                 [7., 9.]]])
    >>> jax.scipy.linalg.det(x)
    Array([-2., 37.], dtype=float32)

[](jax.scipy.linalg.convolution_matrix.html "previous page")

previous

jax.scipy.linalg.convolution_matrix

[](jax.scipy.linalg.dft.html "next page")

next

jax.scipy.linalg.dft

Contents

- [`det()`](#jax.scipy.linalg.det)

By The JAX authors

© Copyright 2024, The JAX Authors.\
