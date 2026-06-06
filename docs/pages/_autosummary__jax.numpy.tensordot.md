- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.tensordot

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.tensordot.rst "Download source file")
-  .pdf

# jax.numpy.tensordot

## Contents

- [`tensordot()`](#jax.numpy.tensordot)

# jax.numpy.tensordot[\#](#jax-numpy-tensordot "Link to this heading")

jax.numpy.tensordot(*a*, *b*, *axes=2*, *\**, *precision=None*, *preferred_element_type=None*, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/tensor_contractions.py#L469-L586)[\#](#jax.numpy.tensordot "Link to this definition")  
Compute the tensor dot product of two N-dimensional arrays.

JAX implementation of [`numpy.linalg.tensordot()`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.tensordot.html#numpy.linalg.tensordot "(in NumPy v2.4)").

Parameters:  
- **a** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – N-dimensional array

- **b** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – M-dimensional array

- **axes** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*Sequence*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]* *\|* [*Sequence*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "(in Python v3.14)")*\[*[*Sequence*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]\]*) – integer or tuple of sequences of integers. If an integer k, then sum over the last k axes of `a` and the first k axes of `b`, in order. If a tuple, then `axes[0]` specifies the axes of `a` and `axes[1]` specifies the axes of `b`.

- **precision** (*None* *\|* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* [*Precision*](../jax.lax.html#jax.lax.Precision "jax._src.lax.lax.Precision") *\|* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*,* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*\]* *\|* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*Precision*](../jax.lax.html#jax.lax.Precision "jax._src.lax.lax.Precision")*,* [*Precision*](../jax.lax.html#jax.lax.Precision "jax._src.lax.lax.Precision")*\]* *\|* [*DotAlgorithm*](../jax.lax.html#jax.lax.DotAlgorithm "jax._src.lax.lax.DotAlgorithm") *\|* [*DotAlgorithmPreset*](../jax.lax.html#jax.lax.DotAlgorithmPreset "jax._src.lax.lax.DotAlgorithmPreset")) – either `None` (default), which means the default precision for the backend, a [`Precision`](../jax.lax.html#jax.lax.Precision "jax.lax.Precision") enum value (`Precision.DEFAULT`, `Precision.HIGH` or `Precision.HIGHEST`) or a tuple of two such values indicating precision of `a` and `b`.

- **preferred_element_type** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* [*type*](https://docs.python.org/3/library/functions.html#type "(in Python v3.14)")*\[*[*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.14)")*\]* *\|* [*dtype*](jax.numpy.dtype.html#jax.numpy.dtype "numpy.dtype") *\|* *SupportsDType* *\|* *None*) – either `None` (default), which means the default accumulation type for the input types, or a datatype, indicating to accumulate results to and return a result with that datatype.

- **out_sharding** ([*NamedSharding*](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding") *\|* *P* *\|* *None*)

Returns:  
array containing the tensor dot product of the inputs

Return type:  
[*Array*](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.einsum()`](jax.numpy.einsum.html#jax.numpy.einsum "jax.numpy.einsum"): NumPy API for more general tensor contractions.

- [`jax.lax.dot_general()`](jax.lax.dot_general.html#jax.lax.dot_general "jax.lax.dot_general"): XLA API for more general tensor contractions.

Examples

    >>> x1 = jnp.arange(24.).reshape(2, 3, 4)
    >>> x2 = jnp.ones((3, 4, 5))
    >>> jnp.tensordot(x1, x2)
    Array([[ 66.,  66.,  66.,  66.,  66.],
           [210., 210., 210., 210., 210.]], dtype=float32)

Equivalent result when specifying the axes as explicit sequences:

    >>> jnp.tensordot(x1, x2, axes=([1, 2], [0, 1]))
    Array([[ 66.,  66.,  66.,  66.,  66.],
           [210., 210., 210., 210., 210.]], dtype=float32)

Equivalent result via [`einsum()`](jax.numpy.einsum.html#jax.numpy.einsum "jax.numpy.einsum"):

    >>> jnp.einsum('ijk,jkm->im', x1, x2)
    Array([[ 66.,  66.,  66.,  66.,  66.],
           [210., 210., 210., 210., 210.]], dtype=float32)

Setting `axes=1` for two-dimensional inputs is equivalent to a matrix multiplication:

    >>> x1 = jnp.array([[1, 2],
    ...                 [3, 4]])
    >>> x2 = jnp.array([[1, 2, 3],
    ...                 [4, 5, 6]])
    >>> jnp.linalg.tensordot(x1, x2, axes=1)
    Array([[ 9, 12, 15],
           [19, 26, 33]], dtype=int32)
    >>> x1 @ x2
    Array([[ 9, 12, 15],
           [19, 26, 33]], dtype=int32)

Setting `axes=0` for one-dimensional inputs is equivalent to [`outer()`](jax.numpy.outer.html#jax.numpy.outer "jax.numpy.outer"):

    >>> x1 = jnp.array([1, 2])
    >>> x2 = jnp.array([1, 2, 3])
    >>> jnp.linalg.tensordot(x1, x2, axes=0)
    Array([[1, 2, 3],
           [2, 4, 6]], dtype=int32)
    >>> jnp.outer(x1, x2)
    Array([[1, 2, 3],
           [2, 4, 6]], dtype=int32)

[](jax.numpy.tanh.html "previous page")

previous

jax.numpy.tanh

[](jax.numpy.tile.html "next page")

next

jax.numpy.tile

Contents

- [`tensordot()`](#jax.numpy.tensordot)

By The JAX authors

© Copyright 2024, The JAX Authors.\
