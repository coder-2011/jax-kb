- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.dot

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.dot.rst "Download source file")
-  .pdf

# jax.numpy.dot

## Contents

- [`dot()`](#jax.numpy.dot)

# jax.numpy.dot[\#](#jax-numpy-dot "Link to this heading")

jax.numpy.dot(*a*, *b*, *\**, *precision=None*, *preferred_element_type=None*, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/tensor_contractions.py#L36-L128)[\#](#jax.numpy.dot "Link to this definition")  
Compute the dot product of two arrays.

JAX implementation of [`numpy.dot()`](https://numpy.org/doc/stable/reference/generated/numpy.dot.html#numpy.dot "(in NumPy v2.4)").

This differs from [`jax.numpy.matmul()`](jax.numpy.matmul.html#jax.numpy.matmul "jax.numpy.matmul") in two respects:

- if either `a` or `b` is a scalar, the result of `dot` is equivalent to [`jax.numpy.multiply()`](jax.numpy.multiply.html#jax.numpy.multiply "jax.numpy.multiply"), while the result of `matmul` is an error.

- if `a` and `b` have more than 2 dimensions, the batch indices are stacked rather than broadcast.

Parameters:  
- **a** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – first input array, of shape `(*a_batch,`` ``N)`.

- **b** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – second input array. Must have shape `(N,)` or `(*b_batch,`` ``N,`` ``M)`.

- **precision** (*None* *\|* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* [*Precision*](../jax.lax.html#jax.lax.Precision "jax._src.lax.lax.Precision") *\|* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*,* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*\]* *\|* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*Precision*](../jax.lax.html#jax.lax.Precision "jax._src.lax.lax.Precision")*,* [*Precision*](../jax.lax.html#jax.lax.Precision "jax._src.lax.lax.Precision")*\]* *\|* [*DotAlgorithm*](../jax.lax.html#jax.lax.DotAlgorithm "jax._src.lax.lax.DotAlgorithm") *\|* [*DotAlgorithmPreset*](../jax.lax.html#jax.lax.DotAlgorithmPreset "jax._src.lax.lax.DotAlgorithmPreset")) – either `None` (default), which means the default precision for the backend, a [`Precision`](../jax.lax.html#jax.lax.Precision "jax.lax.Precision") enum value (`Precision.DEFAULT`, `Precision.HIGH` or `Precision.HIGHEST`) or a tuple of two such values indicating precision of `a` and `b`.

- **preferred_element_type** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* [*type*](https://docs.python.org/3/library/functions.html#type "(in Python v3.14)")*\[*[*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.14)")*\]* *\|* [*dtype*](jax.numpy.dtype.html#jax.numpy.dtype "numpy.dtype") *\|* *SupportsDType* *\|* *None*) – either `None` (default), which means the default accumulation type for the input types, or a datatype, indicating to accumulate results to and return a result with that datatype.

Returns:  
An array containing the dot product of the inputs. Unlike [`matmul()`](jax.numpy.matmul.html#jax.numpy.matmul "jax.numpy.matmul"), the batch dimensions of `a` and `b` are stacked rather than broadcast; that is, the output shape will be `(*a_batch,)` if `b` is one-dimensional, or `(*a_batch,`` ``*b_batch,`` ``M)` if `b` has more than one dimension.

Return type:  
[*Array*](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.matmul()`](jax.numpy.matmul.html#jax.numpy.matmul "jax.numpy.matmul"): broadcasted batched matmul.

- [`jax.lax.dot_general()`](jax.lax.dot_general.html#jax.lax.dot_general "jax.lax.dot_general"): general batched matrix multiplication.

Examples

For scalar inputs, `dot` computes the element-wise product:

    >>> x = jnp.array([1, 2, 3])
    >>> jnp.dot(x, 2)
    Array([2, 4, 6], dtype=int32)

For vector or matrix inputs, `dot` computes the vector or matrix product:

    >>> M = jnp.array([[2, 3, 4],
    ...                [5, 6, 7],
    ...                [8, 9, 0]])
    >>> jnp.dot(M, x)
    Array([20, 38, 26], dtype=int32)
    >>> jnp.dot(M, M)
    Array([[ 51,  60,  29],
           [ 96, 114,  62],
           [ 61,  78,  95]], dtype=int32)

For higher-dimensional matrix products, batch dimensions are stacked, whereas in [`matmul()`](jax.numpy.matmul.html#jax.numpy.matmul "jax.numpy.matmul") they are broadcast. For example:

    >>> a = jnp.zeros((3, 2, 4))
    >>> b = jnp.zeros((3, 4, 1))
    >>> jnp.dot(a, b).shape
    (3, 2, 3, 1)
    >>> jnp.matmul(a, b).shape
    (3, 2, 1)

[](jax.numpy.divmod.html "previous page")

previous

jax.numpy.divmod

[](jax.numpy.double.html "next page")

next

jax.numpy.double

Contents

- [`dot()`](#jax.numpy.dot)

By The JAX authors

© Copyright 2024, The JAX Authors.\
