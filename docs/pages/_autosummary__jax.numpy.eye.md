- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.eye

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.eye.rst "Download source file")
-  .pdf

# jax.numpy.eye

## Contents

- [`eye()`](#jax.numpy.eye)

# jax.numpy.eye[\#](#jax-numpy-eye "Link to this heading")

jax.numpy.eye(*N*, *M=None*, *k=0*, *dtype=None*, *\**, *device=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L5717-L5775)[\#](#jax.numpy.eye "Link to this definition")  
Create a square or rectangular identity matrix

JAX implementation of [`numpy.eye()`](https://numpy.org/doc/stable/reference/generated/numpy.eye.html#numpy.eye "(in NumPy v2.4)").

Parameters:  
- **N** (*DimSize*) – integer specifying the first dimension of the array.

- **M** (*DimSize* *\|* *None*) – optional integer specifying the second dimension of the array; defaults to the same value as `N`.

- **k** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *ArrayLike*) – optional integer specifying the offset of the diagonal. Use positive values for upper diagonals, and negative values for lower diagonals. Default is zero.

- **dtype** (*DTypeLike* *\|* *None*) – optional dtype; defaults to floating point.

- **device** (*xc.Device* *\|* [*Sharding*](../jax.sharding.html#jax.sharding.Sharding "jax.sharding.Sharding") *\|* *None*) – optional [`Device`](jax.Device.html#jax.Device "jax.Device") or [`Sharding`](../jax.sharding.html#jax.sharding.Sharding "jax.sharding.Sharding") to which the created array will be committed.

Returns:  
Identity array of shape `(N,`` ``M)`, or `(N,`` ``N)` if `M` is not specified.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

[`jax.numpy.identity()`](jax.numpy.identity.html#jax.numpy.identity "jax.numpy.identity"): Simpler API for generating square identity matrices.

Examples

A simple 3x3 identity matrix:

    >>> jnp.eye(3)
    Array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]], dtype=float32)

Integer identity matrices with offset diagonals:

    >>> jnp.eye(3, k=1, dtype=int)
    Array([[0, 1, 0],
           [0, 0, 1],
           [0, 0, 0]], dtype=int32)
    >>> jnp.eye(3, k=-1, dtype=int)
    Array([[0, 0, 0],
           [1, 0, 0],
           [0, 1, 0]], dtype=int32)

Non-square identity matrix:

    >>> jnp.eye(3, 5, k=1)
    Array([[0., 1., 0., 0., 0.],
           [0., 0., 1., 0., 0.],
           [0., 0., 0., 1., 0.]], dtype=float32)

[](jax.numpy.extract.html "previous page")

previous

jax.numpy.extract

[](jax.numpy.fabs.html "next page")

next

jax.numpy.fabs

Contents

- [`eye()`](#jax.numpy.eye)

By The JAX authors

© Copyright 2024, The JAX Authors.\
