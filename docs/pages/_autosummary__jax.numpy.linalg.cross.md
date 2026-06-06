- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.linalg.cross

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.linalg.cross.rst "Download source file")
-  .pdf

# jax.numpy.linalg.cross

## Contents

- [`cross()`](#jax.numpy.linalg.cross)

# jax.numpy.linalg.cross[\#](#jax-numpy-linalg-cross "Link to this heading")

jax.numpy.linalg.cross(*x1*, *x2*, */*, *\**, *axis=-1*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/linalg.py#L1531-L1574)[\#](#jax.numpy.linalg.cross "Link to this definition")  
Compute the cross-product of two 3D vectors

JAX implementation of [`numpy.linalg.cross()`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.cross.html#numpy.linalg.cross "(in NumPy v2.4)")

Parameters:  
- **x1** (*ArrayLike*) – N-dimensional array, with `x1.shape[axis]`` ``==`` ``3`

- **x2** (*ArrayLike*) – N-dimensional array, with `x2.shape[axis]`` ``==`` ``3`, and other axes broadcast-compatible with `x1`.

- **axis** – axis along which to take the cross product (default: -1).

Returns:  
array containing the result of the cross-product

See also

[`jax.numpy.cross()`](jax.numpy.cross.html#jax.numpy.cross "jax.numpy.cross"): more flexible cross-product API.

Examples

Showing that \\\hat{x} \times \hat{y} = \hat{z}\\:

    >>> x = jnp.array([1., 0., 0.])
    >>> y = jnp.array([0., 1., 0.])
    >>> jnp.linalg.cross(x, y)
    Array([0., 0., 1.], dtype=float32)

Cross product of \\\hat{x}\\ with all three standard unit vectors, via broadcasting:

    >>> xyz = jnp.eye(3)
    >>> jnp.linalg.cross(x, xyz, axis=-1)
    Array([[ 0.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 0., -1.,  0.]], dtype=float32)

[](jax.numpy.linalg.cond.html "previous page")

previous

jax.numpy.linalg.cond

[](jax.numpy.linalg.det.html "next page")

next

jax.numpy.linalg.det

Contents

- [`cross()`](#jax.numpy.linalg.cross)

By The JAX authors

© Copyright 2024, The JAX Authors.\
