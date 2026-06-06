- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.linalg.det

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.linalg.det.rst "Download source file")
-  .pdf

# jax.numpy.linalg.det

## Contents

- [`det()`](#jax.numpy.linalg.det)

# jax.numpy.linalg.det[\#](#jax-numpy-linalg-det "Link to this heading")

jax.numpy.linalg.det(*a*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/linalg.py#L761-L796)[\#](#jax.numpy.linalg.det "Link to this definition")  
Compute the determinant of an array.

JAX implementation of [`numpy.linalg.det()`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.det.html#numpy.linalg.det "(in NumPy v2.4)").

Parameters:  
**a** (*ArrayLike*) – array of shape `(...,`` ``M,`` ``M)` for which to compute the determinant.

Returns:  
An array of determinants of shape `a.shape[:-2]`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

[`jax.scipy.linalg.det()`](jax.scipy.linalg.det.html#jax.scipy.linalg.det "jax.scipy.linalg.det"): Scipy-style API for determinant.

Examples

    >>> a = jnp.array([[1, 2],
    ...                [3, 4]])
    >>> jnp.linalg.det(a)
    Array(-2., dtype=float32)

[](jax.numpy.linalg.cross.html "previous page")

previous

jax.numpy.linalg.cross

[](jax.numpy.linalg.diagonal.html "next page")

next

jax.numpy.linalg.diagonal

Contents

- [`det()`](#jax.numpy.linalg.det)

By The JAX authors

© Copyright 2024, The JAX Authors.\
