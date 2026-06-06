- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.vecmat

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.vecmat.rst "Download source file")
-  .pdf

# jax.numpy.vecmat

## Contents

- [`vecmat()`](#jax.numpy.vecmat)

# jax.numpy.vecmat[\#](#jax-numpy-vecmat "Link to this heading")

jax.numpy.vecmat(*x1*, *x2*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/tensor_contractions.py#L320-L360)[\#](#jax.numpy.vecmat "Link to this definition")  
Batched conjugate vector-matrix product.

JAX implementation of `numpy.vecmat()`.

Parameters:  
- **x1** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – array of shape `(...,`` ``M)`.

- **x2** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – array of shape `(...,`` ``M,`` ``N)`. Leading dimensions must be broadcast-compatible with leading dimensions of `x1`.

Returns:  
An array of shape `(...,`` ``N)` containing the batched conjugate vector-matrix product.

Return type:  
[*Array*](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.linalg.vecdot()`](jax.numpy.linalg.vecdot.html#jax.numpy.linalg.vecdot "jax.numpy.linalg.vecdot"): batched vector product.

- [`jax.numpy.matvec()`](jax.numpy.matvec.html#jax.numpy.matvec "jax.numpy.matvec"): matrix-vector product.

- [`jax.numpy.matmul()`](jax.numpy.matmul.html#jax.numpy.matmul "jax.numpy.matmul"): general matrix multiplication.

Examples

Simple vector-matrix product:

    >>> x1 = jnp.array([[1, 2, 3]])
    >>> x2 = jnp.array([[4, 5],
    ...                 [6, 7],
    ...                 [8, 9]])
    >>> jnp.vecmat(x1, x2)
    Array([[40, 46]], dtype=int32)

Batched vector-matrix product:

    >>> x1 = jnp.array([[1, 2, 3],
    ...                 [4, 5, 6]])
    >>> jnp.vecmat(x1, x2)
    Array([[ 40,  46],
           [ 94, 109]], dtype=int32)

[](jax.numpy.vecdot.html "previous page")

previous

jax.numpy.vecdot

[](jax.numpy.vectorize.html "next page")

next

jax.numpy.vectorize

Contents

- [`vecmat()`](#jax.numpy.vecmat)

By The JAX authors

© Copyright 2024, The JAX Authors.\
