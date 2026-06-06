- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.inner

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.inner.rst "Download source file")
-  .pdf

# jax.numpy.inner

## Contents

- [`inner()`](#jax.numpy.inner)

# jax.numpy.inner[\#](#jax-numpy-inner "Link to this heading")

jax.numpy.inner(*a*, *b*, *\**, *precision=None*, *preferred_element_type=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/tensor_contractions.py#L589-L645)[\#](#jax.numpy.inner "Link to this definition")  
Compute the inner product of two arrays.

JAX implementation of [`numpy.inner()`](https://numpy.org/doc/stable/reference/generated/numpy.inner.html#numpy.inner "(in NumPy v2.4)").

Unlike [`jax.numpy.matmul()`](jax.numpy.matmul.html#jax.numpy.matmul "jax.numpy.matmul") or [`jax.numpy.dot()`](jax.numpy.dot.html#jax.numpy.dot "jax.numpy.dot"), this always performs a contraction along the last dimension of each input.

Parameters:  
- **a** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – array of shape `(...,`` ``N)`

- **b** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – array of shape `(...,`` ``N)`

- **precision** (*None* *\|* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* [*Precision*](../jax.lax.html#jax.lax.Precision "jax._src.lax.lax.Precision") *\|* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*,* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*\]* *\|* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*Precision*](../jax.lax.html#jax.lax.Precision "jax._src.lax.lax.Precision")*,* [*Precision*](../jax.lax.html#jax.lax.Precision "jax._src.lax.lax.Precision")*\]* *\|* [*DotAlgorithm*](../jax.lax.html#jax.lax.DotAlgorithm "jax._src.lax.lax.DotAlgorithm") *\|* [*DotAlgorithmPreset*](../jax.lax.html#jax.lax.DotAlgorithmPreset "jax._src.lax.lax.DotAlgorithmPreset")) – either `None` (default), which means the default precision for the backend, a [`Precision`](../jax.lax.html#jax.lax.Precision "jax.lax.Precision") enum value (`Precision.DEFAULT`, `Precision.HIGH` or `Precision.HIGHEST`) or a tuple of two such values indicating precision of `a` and `b`.

- **preferred_element_type** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* [*type*](https://docs.python.org/3/library/functions.html#type "(in Python v3.14)")*\[*[*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.14)")*\]* *\|* [*dtype*](jax.numpy.dtype.html#jax.numpy.dtype "numpy.dtype") *\|* *SupportsDType* *\|* *None*) – either `None` (default), which means the default accumulation type for the input types, or a datatype, indicating to accumulate results to and return a result with that datatype.

Returns:  
array of shape `(*a.shape[:-1],`` ``*b.shape[:-1])` containing the batched vector product of the inputs.

Return type:  
[*Array*](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.vecdot()`](jax.numpy.vecdot.html#jax.numpy.vecdot "jax.numpy.vecdot"): conjugate multiplication along a specified axis.

- [`jax.numpy.tensordot()`](jax.numpy.tensordot.html#jax.numpy.tensordot "jax.numpy.tensordot"): general tensor multiplication.

- [`jax.numpy.matmul()`](jax.numpy.matmul.html#jax.numpy.matmul "jax.numpy.matmul"): general batched matrix & vector multiplication.

Examples

For 1D inputs, this implements standard (non-conjugate) vector multiplication:

    >>> a = jnp.array([1j, 3j, 4j])
    >>> b = jnp.array([4., 2., 5.])
    >>> jnp.inner(a, b)
    Array(0.+30.j, dtype=complex64)

For multi-dimensional inputs, batch dimensions are stacked rather than broadcast:

    >>> a = jnp.ones((2, 3))
    >>> b = jnp.ones((5, 3))
    >>> jnp.inner(a, b).shape
    (2, 5)

[](jax.numpy.inexact.html "previous page")

previous

jax.numpy.inexact

[](jax.numpy.insert.html "next page")

next

jax.numpy.insert

Contents

- [`inner()`](#jax.numpy.inner)

By The JAX authors

© Copyright 2024, The JAX Authors.\
