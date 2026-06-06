- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.matvec

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.matvec.rst "Download source file")
-  .pdf

# jax.numpy.matvec

## Contents

- [`matvec()`](#jax.numpy.matvec)

# jax.numpy.matvec[\#](#jax-numpy-matvec "Link to this heading")

jax.numpy.matvec(*x1*, *x2*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/tensor_contractions.py#L279-L318)[\#](#jax.numpy.matvec "Link to this definition")  
Batched matrix-vector product.

JAX implementation of `numpy.matvec()`.

Parameters:  
- **x1** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – array of shape `(...,`` ``M,`` ``N)`

- **x2** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – array of shape `(...,`` ``N)`. Leading dimensions must be broadcast-compatible with leading dimensions of `x1`.

Returns:  
An array of shape `(...,`` ``M)` containing the batched matrix-vector product.

Return type:  
[*Array*](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.linalg.vecdot()`](jax.numpy.linalg.vecdot.html#jax.numpy.linalg.vecdot "jax.numpy.linalg.vecdot"): batched vector product.

- [`jax.numpy.vecmat()`](jax.numpy.vecmat.html#jax.numpy.vecmat "jax.numpy.vecmat"): vector-matrix product.

- [`jax.numpy.matmul()`](jax.numpy.matmul.html#jax.numpy.matmul "jax.numpy.matmul"): general matrix multiplication.

Examples

Simple matrix-vector product:

    >>> x1 = jnp.array([[1, 2, 3],
    ...                 [4, 5, 6]])
    >>> x2 = jnp.array([7, 8, 9])
    >>> jnp.matvec(x1, x2)
    Array([ 50, 122], dtype=int32)

Batched matrix-vector product:

    >>> x2 = jnp.array([[7, 8, 9],
    ...                 [5, 6, 7]])
    >>> jnp.matvec(x1, x2)
    Array([[ 50, 122],
           [ 38,  92]], dtype=int32)

[](jax.numpy.matrix_transpose.html "previous page")

previous

jax.numpy.matrix_transpose

[](jax.numpy.max.html "next page")

next

jax.numpy.max

Contents

- [`matvec()`](#jax.numpy.matvec)

By The JAX authors

© Copyright 2024, The JAX Authors.\
