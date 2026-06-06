- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.linalg.matrix_transpose

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.linalg.matrix_transpose.rst "Download source file")
-  .pdf

# jax.numpy.linalg.matrix_transpose

## Contents

- [`matrix_transpose()`](#jax.numpy.linalg.matrix_transpose)

# jax.numpy.linalg.matrix_transpose[\#](#jax-numpy-linalg-matrix-transpose "Link to this heading")

jax.numpy.linalg.matrix_transpose(*x*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/linalg.py#L1637-L1690)[\#](#jax.numpy.linalg.matrix_transpose "Link to this definition")  
Transpose a matrix or stack of matrices.

JAX implementation of [`numpy.linalg.matrix_transpose()`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_transpose.html#numpy.linalg.matrix_transpose "(in NumPy v2.4)").

Parameters:  
**x** (*ArrayLike*) – array of shape `(...,`` ``M,`` ``N)`

Returns:  
array of shape `(...,`` ``N,`` ``M)` containing the matrix transpose of `x`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

[`jax.numpy.transpose()`](jax.numpy.transpose.html#jax.numpy.transpose "jax.numpy.transpose"): more general transpose operation.

Examples

Transpose of a single matrix:

    >>> x = jnp.array([[1, 2, 3],
    ...                [4, 5, 6]])
    >>> jnp.linalg.matrix_transpose(x)
    Array([[1, 4],
           [2, 5],
           [3, 6]], dtype=int32)

Transpose of a stack of matrices:

    >>> x = jnp.array([[[1, 2],
    ...                 [3, 4]],
    ...                [[5, 6],
    ...                 [7, 8]]])
    >>> jnp.linalg.matrix_transpose(x)
    Array([[[1, 3],
            [2, 4]],

           [[5, 7],
            [6, 8]]], dtype=int32)

For convenience, the same computation can be done via the [`mT`](jax.Array.mT.html#jax.Array.mT "jax.Array.mT") property of JAX array objects:

    >>> x.mT
    Array([[[1, 3],
            [2, 4]],

           [[5, 7],
            [6, 8]]], dtype=int32)

[](jax.numpy.linalg.matrix_rank.html "previous page")

previous

jax.numpy.linalg.matrix_rank

[](jax.numpy.linalg.multi_dot.html "next page")

next

jax.numpy.linalg.multi_dot

Contents

- [`matrix_transpose()`](#jax.numpy.linalg.matrix_transpose)

By The JAX authors

© Copyright 2024, The JAX Authors.\
