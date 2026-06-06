- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.linalg.tensordot

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.linalg.tensordot.rst "Download source file")
-  .pdf

# jax.numpy.linalg.tensordot

## Contents

- [`tensordot()`](#jax.numpy.linalg.tensordot)

# jax.numpy.linalg.tensordot[\#](#jax-numpy-linalg-tensordot "Link to this heading")

jax.numpy.linalg.tensordot(*x1*, *x2*, */*, *\**, *axes=2*, *precision=None*, *preferred_element_type=None*, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/linalg.py#L1871-L1953)[\#](#jax.numpy.linalg.tensordot "Link to this definition")  
Compute the tensor dot product of two N-dimensional arrays.

JAX implementation of [`numpy.linalg.tensordot()`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.tensordot.html#numpy.linalg.tensordot "(in NumPy v2.4)").

Parameters:  
- **x1** (*ArrayLike*) – N-dimensional array

- **x2** (*ArrayLike*) – M-dimensional array

- **axes** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\],* *Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]\]*) – integer or tuple of sequences of integers. If an integer k, then sum over the last k axes of `x1` and the first k axes of `x2`, in order. If a tuple, then `axes[0]` specifies the axes of `x1` and `axes[1]` specifies the axes of `x2`.

- **precision** (*lax.PrecisionLike*) – either `None` (default), which means the default precision for the backend, a [`Precision`](../jax.lax.html#jax.lax.Precision "jax.lax.Precision") enum value (`Precision.DEFAULT`, `Precision.HIGH` or `Precision.HIGHEST`) or a tuple of two such values indicating precision of `x1` and `x2`.

- **preferred_element_type** (*DTypeLike* *\|* *None*) – either `None` (default), which means the default accumulation type for the input types, or a datatype, indicating to accumulate results to and return a result with that datatype.

- **out_sharding** ([*NamedSharding*](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding") *\|* *P* *\|* *None*)

Returns:  
array containing the tensor dot product of the inputs

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.tensordot()`](jax.numpy.tensordot.html#jax.numpy.tensordot "jax.numpy.tensordot"): equivalent API in the [`jax.numpy`](../jax.numpy.html#module-jax.numpy "jax.numpy") namespace.

- [`jax.numpy.einsum()`](jax.numpy.einsum.html#jax.numpy.einsum "jax.numpy.einsum"): NumPy API for more general tensor contractions.

- [`jax.lax.dot_general()`](jax.lax.dot_general.html#jax.lax.dot_general "jax.lax.dot_general"): XLA API for more general tensor contractions.

Examples

    >>> x1 = jnp.arange(24.).reshape(2, 3, 4)
    >>> x2 = jnp.ones((3, 4, 5))
    >>> jnp.linalg.tensordot(x1, x2)
    Array([[ 66.,  66.,  66.,  66.,  66.],
           [210., 210., 210., 210., 210.]], dtype=float32)

Equivalent result when specifying the axes as explicit sequences:

    >>> jnp.linalg.tensordot(x1, x2, axes=([1, 2], [0, 1]))
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

Setting `axes=0` for one-dimensional inputs is equivalent to [`jax.numpy.linalg.outer()`](jax.numpy.linalg.outer.html#jax.numpy.linalg.outer "jax.numpy.linalg.outer"):

    >>> x1 = jnp.array([1, 2])
    >>> x2 = jnp.array([1, 2, 3])
    >>> jnp.linalg.tensordot(x1, x2, axes=0)
    Array([[1, 2, 3],
           [2, 4, 6]], dtype=int32)
    >>> jnp.linalg.outer(x1, x2)
    Array([[1, 2, 3],
           [2, 4, 6]], dtype=int32)

[](jax.numpy.linalg.svdvals.html "previous page")

previous

jax.numpy.linalg.svdvals

[](jax.numpy.linalg.tensorinv.html "next page")

next

jax.numpy.linalg.tensorinv

Contents

- [`tensordot()`](#jax.numpy.linalg.tensordot)

By The JAX authors

© Copyright 2024, The JAX Authors.\
