- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.linalg.matmul

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.linalg.matmul.rst "Download source file")
-  .pdf

# jax.numpy.linalg.matmul

## Contents

- [`matmul()`](#jax.numpy.linalg.matmul)

# jax.numpy.linalg.matmul[\#](#jax-numpy-linalg-matmul "Link to this heading")

jax.numpy.linalg.matmul(*x1*, *x2*, */*, *\**, *precision=None*, *preferred_element_type=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/linalg.py#L1810-L1869)[\#](#jax.numpy.linalg.matmul "Link to this definition")  
Perform a matrix multiplication.

JAX implementation of [`numpy.linalg.matmul()`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.matmul.html#numpy.linalg.matmul "(in NumPy v2.4)").

Parameters:  
- **x1** (*ArrayLike*) – first input array, of shape `(...,`` ``N)`.

- **x2** (*ArrayLike*) – second input array. Must have shape `(N,)` or `(...,`` ``N,`` ``M)`. In the multi-dimensional case, leading dimensions must be broadcast-compatible with the leading dimensions of `x1`.

- **precision** (*lax.PrecisionLike*) – either `None` (default), which means the default precision for the backend, a [`Precision`](../jax.lax.html#jax.lax.Precision "jax.lax.Precision") enum value (`Precision.DEFAULT`, `Precision.HIGH` or `Precision.HIGHEST`) or a tuple of two such values indicating precision of `x1` and `x2`.

- **preferred_element_type** (*DTypeLike* *\|* *None*) – either `None` (default), which means the default accumulation type for the input types, or a datatype, indicating to accumulate results to and return a result with that datatype.

Returns:  
array containing the matrix product of the inputs. Shape is `x1.shape[:-1]` if `x2.ndim`` ``==`` ``1`, otherwise the shape is `(...,`` ``M)`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

[`jax.numpy.matmul()`](jax.numpy.matmul.html#jax.numpy.matmul "jax.numpy.matmul"): NumPy API for this function. [`jax.numpy.linalg.vecdot()`](jax.numpy.linalg.vecdot.html#jax.numpy.linalg.vecdot "jax.numpy.linalg.vecdot"): batched vector product. [`jax.numpy.linalg.tensordot()`](jax.numpy.linalg.tensordot.html#jax.numpy.linalg.tensordot "jax.numpy.linalg.tensordot"): batched tensor product.

Examples

Vector dot products:

    >>> x1 = jnp.array([1, 2, 3])
    >>> x2 = jnp.array([4, 5, 6])
    >>> jnp.linalg.matmul(x1, x2)
    Array(32, dtype=int32)

Matrix dot product:

    >>> x1 = jnp.array([[1, 2, 3],
    ...                 [4, 5, 6]])
    >>> x2 = jnp.array([[1, 2],
    ...                 [3, 4],
    ...                 [5, 6]])
    >>> jnp.linalg.matmul(x1, x2)
    Array([[22, 28],
           [49, 64]], dtype=int32)

For convenience, in all cases you can do the same computation using the `@` operator:

    >>> x1 @ x2
    Array([[22, 28],
           [49, 64]], dtype=int32)

[](jax.numpy.linalg.lstsq.html "previous page")

previous

jax.numpy.linalg.lstsq

[](jax.numpy.linalg.matrix_norm.html "next page")

next

jax.numpy.linalg.matrix_norm

Contents

- [`matmul()`](#jax.numpy.linalg.matmul)

By The JAX authors

© Copyright 2024, The JAX Authors.\
