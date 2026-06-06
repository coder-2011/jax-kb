- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.matmul

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.matmul.rst "Download source file")
-  .pdf

# jax.numpy.matmul

## Contents

- [`matmul()`](#jax.numpy.matmul)

# jax.numpy.matmul[\#](#jax-numpy-matmul "Link to this heading")

jax.numpy.matmul(*a*, *b*, *\**, *precision=None*, *preferred_element_type=None*, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/tensor_contractions.py#L130-L277)[\#](#jax.numpy.matmul "Link to this definition")  
Perform a matrix multiplication.

JAX implementation of `numpy.matmul()`.

Parameters:  
- **a** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – first input array, of shape `(N,)` or `(...,`` ``K,`` ``N)`.

- **b** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – second input array. Must have shape `(N,)` or `(...,`` ``N,`` ``M)`. In the multi-dimensional case, leading dimensions must be broadcast-compatible with the leading dimensions of `a`.

- **precision** (*None* *\|* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* [*Precision*](../jax.lax.html#jax.lax.Precision "jax._src.lax.lax.Precision") *\|* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*,* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*\]* *\|* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*Precision*](../jax.lax.html#jax.lax.Precision "jax._src.lax.lax.Precision")*,* [*Precision*](../jax.lax.html#jax.lax.Precision "jax._src.lax.lax.Precision")*\]* *\|* [*DotAlgorithm*](../jax.lax.html#jax.lax.DotAlgorithm "jax._src.lax.lax.DotAlgorithm") *\|* [*DotAlgorithmPreset*](../jax.lax.html#jax.lax.DotAlgorithmPreset "jax._src.lax.lax.DotAlgorithmPreset")) – either `None` (default), which means the default precision for the backend, a [`Precision`](../jax.lax.html#jax.lax.Precision "jax.lax.Precision") enum value (`Precision.DEFAULT`, `Precision.HIGH` or `Precision.HIGHEST`) or a tuple of two such values indicating precision of `a` and `b`.

- **preferred_element_type** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* [*type*](https://docs.python.org/3/library/functions.html#type "(in Python v3.14)")*\[*[*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.14)")*\]* *\|* [*dtype*](jax.numpy.dtype.html#jax.numpy.dtype "numpy.dtype") *\|* *SupportsDType* *\|* *None*) – either `None` (default), which means the default accumulation type for the input types, or a datatype, indicating to accumulate results to and return a result with that datatype.

- **out_sharding** ([*NamedSharding*](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding") *\|* *P* *\|* *None*)

Returns:  
array containing the matrix product of the inputs. Shape is `a.shape[:-1]` if `b.ndim`` ``==`` ``1`, otherwise the shape is `(...,`` ``K,`` ``M)`, where leading dimensions of `a` and `b` are broadcast together.

Return type:  
[*Array*](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.linalg.vecdot()`](jax.numpy.linalg.vecdot.html#jax.numpy.linalg.vecdot "jax.numpy.linalg.vecdot"): batched vector product.

- [`jax.numpy.linalg.tensordot()`](jax.numpy.linalg.tensordot.html#jax.numpy.linalg.tensordot "jax.numpy.linalg.tensordot"): batched tensor product.

- [`jax.lax.dot_general()`](jax.lax.dot_general.html#jax.lax.dot_general "jax.lax.dot_general"): general N-dimensional batched dot product.

Examples

Vector dot products:

    >>> a = jnp.array([1, 2, 3])
    >>> b = jnp.array([4, 5, 6])
    >>> jnp.matmul(a, b)
    Array(32, dtype=int32)

Matrix dot product:

    >>> a = jnp.array([[1, 2, 3],
    ...                [4, 5, 6]])
    >>> b = jnp.array([[1, 2],
    ...                [3, 4],
    ...                [5, 6]])
    >>> jnp.matmul(a, b)
    Array([[22, 28],
           [49, 64]], dtype=int32)

For convenience, in all cases you can do the same computation using the `@` operator:

    >>> a @ b
    Array([[22, 28],
           [49, 64]], dtype=int32)

[](jax.numpy.mask_indices.html "previous page")

previous

jax.numpy.mask_indices

[](jax.numpy.matrix_transpose.html "next page")

next

jax.numpy.matrix_transpose

Contents

- [`matmul()`](#jax.numpy.matmul)

By The JAX authors

© Copyright 2024, The JAX Authors.\
