- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.vecdot

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.vecdot.rst "Download source file")
-  .pdf

# jax.numpy.vecdot

## Contents

- [`vecdot()`](#jax.numpy.vecdot)

# jax.numpy.vecdot[\#](#jax-numpy-vecdot "Link to this heading")

jax.numpy.vecdot(*x1*, *x2*, */*, *\**, *axis=-1*, *precision=None*, *preferred_element_type=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/tensor_contractions.py#L411-L467)[\#](#jax.numpy.vecdot "Link to this definition")  
Perform a conjugate multiplication of two batched vectors.

JAX implementation of `numpy.vecdot()`.

Parameters:  
- **a** – left-hand side array.

- **b** – right-hand side array. Size of `b[axis]` must match size of `a[axis]`, and remaining dimensions must be broadcast-compatible.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – axis along which to compute the dot product (default: -1)

- **precision** (*None* *\|* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* [*Precision*](../jax.lax.html#jax.lax.Precision "jax._src.lax.lax.Precision") *\|* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*,* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*\]* *\|* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*Precision*](../jax.lax.html#jax.lax.Precision "jax._src.lax.lax.Precision")*,* [*Precision*](../jax.lax.html#jax.lax.Precision "jax._src.lax.lax.Precision")*\]* *\|* [*DotAlgorithm*](../jax.lax.html#jax.lax.DotAlgorithm "jax._src.lax.lax.DotAlgorithm") *\|* [*DotAlgorithmPreset*](../jax.lax.html#jax.lax.DotAlgorithmPreset "jax._src.lax.lax.DotAlgorithmPreset")) – either `None` (default), which means the default precision for the backend, a [`Precision`](../jax.lax.html#jax.lax.Precision "jax.lax.Precision") enum value (`Precision.DEFAULT`, `Precision.HIGH` or `Precision.HIGHEST`) or a tuple of two such values indicating precision of `a` and `b`.

- **preferred_element_type** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* [*type*](https://docs.python.org/3/library/functions.html#type "(in Python v3.14)")*\[*[*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.14)")*\]* *\|* [*dtype*](jax.numpy.dtype.html#jax.numpy.dtype "numpy.dtype") *\|* *SupportsDType* *\|* *None*) – either `None` (default), which means the default accumulation type for the input types, or a datatype, indicating to accumulate results to and return a result with that datatype.

- **x1** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)"))

- **x2** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)"))

Returns:  
array containing the conjugate dot product of `a` and `b` along `axis`. The non-contracted dimensions are broadcast together.

Return type:  
[*Array*](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.vdot()`](jax.numpy.vdot.html#jax.numpy.vdot "jax.numpy.vdot"): flattened vector product.

- [`jax.numpy.vecmat()`](jax.numpy.vecmat.html#jax.numpy.vecmat "jax.numpy.vecmat"): vector-matrix product.

- [`jax.numpy.matmul()`](jax.numpy.matmul.html#jax.numpy.matmul "jax.numpy.matmul"): general matrix multiplication.

- [`jax.lax.dot_general()`](jax.lax.dot_general.html#jax.lax.dot_general "jax.lax.dot_general"): general N-dimensional batched dot product.

Examples

Vector conjugate-dot product of two 1D arrays:

    >>> a = jnp.array([1j, 2j, 3j])
    >>> b = jnp.array([4., 5., 6.])
    >>> jnp.linalg.vecdot(a, b)
    Array(0.-32.j, dtype=complex64)

Batched vector dot product of two 2D arrays:

    >>> a = jnp.array([[1, 2, 3],
    ...                [4, 5, 6]])
    >>> b = jnp.array([[2, 3, 4]])
    >>> jnp.linalg.vecdot(a, b, axis=-1)
    Array([20, 47], dtype=int32)

[](jax.numpy.vdot.html "previous page")

previous

jax.numpy.vdot

[](jax.numpy.vecmat.html "next page")

next

jax.numpy.vecmat

Contents

- [`vecdot()`](#jax.numpy.vecdot)

By The JAX authors

© Copyright 2024, The JAX Authors.\
