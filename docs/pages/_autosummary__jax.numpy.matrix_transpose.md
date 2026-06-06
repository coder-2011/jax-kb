- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.matrix_transpose

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.matrix_transpose.rst "Download source file")
-  .pdf

# jax.numpy.matrix_transpose

## Contents

- [`matrix_transpose()`](#jax.numpy.matrix_transpose)

# jax.numpy.matrix_transpose[\#](#jax-numpy-matrix-transpose "Link to this heading")

jax.numpy.matrix_transpose(*x*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L1197-L1249)[\#](#jax.numpy.matrix_transpose "Link to this definition")  
Transpose the last two dimensions of an array.

JAX implementation of [`numpy.matrix_transpose()`](https://numpy.org/doc/stable/reference/generated/numpy.matrix_transpose.html#numpy.matrix_transpose "(in NumPy v2.4)"), implemented in terms of [`jax.lax.transpose()`](jax.lax.transpose.html#jax.lax.transpose "jax.lax.transpose").

Parameters:  
**x** (*ArrayLike*) – input array, Must have `x.ndim`` ``>=`` ``2`

Returns:  
matrix-transposed copy of the array.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.Array.mT`](jax.Array.mT.html#jax.Array.mT "jax.Array.mT"): same operation accessed via an [`Array()`](jax.Array.html#jax.Array "jax.Array") property.

- [`jax.numpy.transpose()`](jax.numpy.transpose.html#jax.numpy.transpose "jax.numpy.transpose"): general multi-axis transpose

Note

Unlike [`numpy.matrix_transpose()`](https://numpy.org/doc/stable/reference/generated/numpy.matrix_transpose.html#numpy.matrix_transpose "(in NumPy v2.4)"), [`jax.numpy.matrix_transpose()`](#jax.numpy.matrix_transpose "jax.numpy.matrix_transpose") will return a copy rather than a view of the input array. However, under JIT, the compiler will optimize-away such copies when possible, so this doesn’t have performance impacts in practice.

Examples

Here is a 2x2x2 matrix representing a batched 2x2 matrix:

    >>> x = jnp.array([[[1, 2],
    ...                 [3, 4]],
    ...                [[5, 6],
    ...                 [7, 8]]])
    >>> jnp.matrix_transpose(x)
    Array([[[1, 3],
            [2, 4]],

           [[5, 7],
            [6, 8]]], dtype=int32)

For convenience, you can perform the same transpose via the [`mT`](jax.Array.mT.html#jax.Array.mT "jax.Array.mT") property of [`jax.Array`](jax.Array.html#jax.Array "jax.Array"):

    >>> x.mT
    Array([[[1, 3],
            [2, 4]],

           [[5, 7],
            [6, 8]]], dtype=int32)

[](jax.numpy.matmul.html "previous page")

previous

jax.numpy.matmul

[](jax.numpy.matvec.html "next page")

next

jax.numpy.matvec

Contents

- [`matrix_transpose()`](#jax.numpy.matrix_transpose)

By The JAX authors

© Copyright 2024, The JAX Authors.\
