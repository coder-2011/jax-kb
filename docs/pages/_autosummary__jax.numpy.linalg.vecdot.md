- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.linalg.vecdot

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.linalg.vecdot.rst "Download source file")
-  .pdf

# jax.numpy.linalg.vecdot

## Contents

- [`vecdot()`](#jax.numpy.linalg.vecdot)

# jax.numpy.linalg.vecdot[\#](#jax-numpy-linalg-vecdot "Link to this heading")

jax.numpy.linalg.vecdot(*x1*, *x2*, */*, *\**, *axis=-1*, *precision=None*, *preferred_element_type=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/linalg.py#L1759-L1808)[\#](#jax.numpy.linalg.vecdot "Link to this definition")  
Compute the (batched) vector conjugate dot product of two arrays.

JAX implementation of [`numpy.linalg.vecdot()`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.vecdot.html#numpy.linalg.vecdot "(in NumPy v2.4)").

Parameters:  
- **x1** (*ArrayLike*) – left-hand side array.

- **x2** (*ArrayLike*) – right-hand side array. Size of `x2[axis]` must match size of `x1[axis]`, and remaining dimensions must be broadcast-compatible.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – axis along which to compute the dot product (default: -1)

- **precision** (*lax.PrecisionLike*) – either `None` (default), which means the default precision for the backend, a [`Precision`](../jax.lax.html#jax.lax.Precision "jax.lax.Precision") enum value (`Precision.DEFAULT`, `Precision.HIGH` or `Precision.HIGHEST`) or a tuple of two such values indicating precision of `x1` and `x2`.

- **preferred_element_type** (*DTypeLike* *\|* *None*) – either `None` (default), which means the default accumulation type for the input types, or a datatype, indicating to accumulate results to and return a result with that datatype.

Returns:  
array containing the conjugate dot product of `x1` and `x2` along `axis`. The non-contracted dimensions are broadcast together.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.vecdot()`](jax.numpy.vecdot.html#jax.numpy.vecdot "jax.numpy.vecdot"): similar API in the `jax.numpy` namespace.

- [`jax.numpy.linalg.matmul()`](jax.numpy.linalg.matmul.html#jax.numpy.linalg.matmul "jax.numpy.linalg.matmul"): matrix multiplication.

- [`jax.numpy.linalg.tensordot()`](jax.numpy.linalg.tensordot.html#jax.numpy.linalg.tensordot "jax.numpy.linalg.tensordot"): general tensor dot product.

Examples

Vector dot product of two 1D arrays:

    >>> x1 = jnp.array([1, 2, 3])
    >>> x2 = jnp.array([4, 5, 6])
    >>> jnp.linalg.vecdot(x1, x2)
    Array(32, dtype=int32)

Batched vector dot product of two 2D arrays:

    >>> x1 = jnp.array([[1, 2, 3],
    ...                 [4, 5, 6]])
    >>> x2 = jnp.array([[2, 3, 4]])
    >>> jnp.linalg.vecdot(x1, x2, axis=-1)
    Array([20, 47], dtype=int32)

[](jax.numpy.linalg.vector_norm.html "previous page")

previous

jax.numpy.linalg.vector_norm

[](../jax.scipy.html "next page")

next

`jax.scipy` module

Contents

- [`vecdot()`](#jax.numpy.linalg.vecdot)

By The JAX authors

© Copyright 2024, The JAX Authors.\
