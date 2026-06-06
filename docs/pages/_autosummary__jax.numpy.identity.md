- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.identity

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.identity.rst "Download source file")
-  .pdf

# jax.numpy.identity

## Contents

- [`identity()`](#jax.numpy.identity)

# jax.numpy.identity[\#](#jax-numpy-identity "Link to this heading")

jax.numpy.identity(*n*, *dtype=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L5797-L5830)[\#](#jax.numpy.identity "Link to this definition")  
Create a square identity matrix

JAX implementation of [`numpy.identity()`](https://numpy.org/doc/stable/reference/generated/numpy.identity.html#numpy.identity "(in NumPy v2.4)").

Parameters:  
- **n** (*DimSize*) – integer specifying the size of each array dimension.

- **dtype** (*DTypeLike* *\|* *None*) – optional dtype; defaults to floating point.

Returns:  
Identity array of shape `(n,`` ``n)`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

[`jax.numpy.eye()`](jax.numpy.eye.html#jax.numpy.eye "jax.numpy.eye"): non-square and/or offset identity matrices.

Examples

A simple 3x3 identity matrix:

    >>> jnp.identity(3)
    Array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]], dtype=float32)

A 2x2 integer identity matrix:

    >>> jnp.identity(2, dtype=int)
    Array([[1, 0],
           [0, 1]], dtype=int32)

[](jax.numpy.i0.html "previous page")

previous

jax.numpy.i0

[](jax.numpy.iinfo.html "next page")

next

jax.numpy.iinfo

Contents

- [`identity()`](#jax.numpy.identity)

By The JAX authors

© Copyright 2024, The JAX Authors.\
