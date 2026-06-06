- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.vdot

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.vdot.rst "Download source file")
-  .pdf

# jax.numpy.vdot

## Contents

- [`vdot()`](#jax.numpy.vdot)

# jax.numpy.vdot[\#](#jax-numpy-vdot "Link to this heading")

jax.numpy.vdot(*a*, *b*, *\**, *precision=None*, *preferred_element_type=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/tensor_contractions.py#L362-L409)[\#](#jax.numpy.vdot "Link to this definition")  
Perform a conjugate multiplication of two 1D vectors.

JAX implementation of [`numpy.vdot()`](https://numpy.org/doc/stable/reference/generated/numpy.vdot.html#numpy.vdot "(in NumPy v2.4)").

Parameters:  
- **a** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – first input array, if not 1D it will be flattened.

- **b** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – second input array, if not 1D it will be flattened. Must have `a.size`` ``==`` ``b.size`.

- **precision** (*None* *\|* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* [*Precision*](../jax.lax.html#jax.lax.Precision "jax._src.lax.lax.Precision") *\|* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*,* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*\]* *\|* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*Precision*](../jax.lax.html#jax.lax.Precision "jax._src.lax.lax.Precision")*,* [*Precision*](../jax.lax.html#jax.lax.Precision "jax._src.lax.lax.Precision")*\]* *\|* [*DotAlgorithm*](../jax.lax.html#jax.lax.DotAlgorithm "jax._src.lax.lax.DotAlgorithm") *\|* [*DotAlgorithmPreset*](../jax.lax.html#jax.lax.DotAlgorithmPreset "jax._src.lax.lax.DotAlgorithmPreset")) – either `None` (default), which means the default precision for the backend, a [`Precision`](../jax.lax.html#jax.lax.Precision "jax.lax.Precision") enum value (`Precision.DEFAULT`, `Precision.HIGH` or `Precision.HIGHEST`) or a tuple of two such values indicating precision of `a` and `b`.

- **preferred_element_type** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* [*type*](https://docs.python.org/3/library/functions.html#type "(in Python v3.14)")*\[*[*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.14)")*\]* *\|* [*dtype*](jax.numpy.dtype.html#jax.numpy.dtype "numpy.dtype") *\|* *SupportsDType* *\|* *None*) – either `None` (default), which means the default accumulation type for the input types, or a datatype, indicating to accumulate results to and return a result with that datatype.

Returns:  
Scalar array (shape `()`) containing the conjugate vector product of the inputs.

Return type:  
[*Array*](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.vecdot()`](jax.numpy.vecdot.html#jax.numpy.vecdot "jax.numpy.vecdot"): batched vector product.

- [`jax.numpy.matmul()`](jax.numpy.matmul.html#jax.numpy.matmul "jax.numpy.matmul"): general matrix multiplication.

- [`jax.lax.dot_general()`](jax.lax.dot_general.html#jax.lax.dot_general "jax.lax.dot_general"): general N-dimensional batched dot product.

Examples

    >>> x = jnp.array([1j, 2j, 3j])
    >>> y = jnp.array([1., 2., 3.])
    >>> jnp.vdot(x, y)
    Array(0.-14.j, dtype=complex64)

Note the difference between this and [`dot()`](jax.numpy.dot.html#jax.numpy.dot "jax.numpy.dot"), which does not conjugate the first input when complex:

    >>> jnp.dot(x, y)
    Array(0.+14.j, dtype=complex64)

[](jax.numpy.var.html "previous page")

previous

jax.numpy.var

[](jax.numpy.vecdot.html "next page")

next

jax.numpy.vecdot

Contents

- [`vdot()`](#jax.numpy.vdot)

By The JAX authors

© Copyright 2024, The JAX Authors.\
