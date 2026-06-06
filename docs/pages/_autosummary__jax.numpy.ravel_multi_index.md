- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.ravel_multi_index

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.ravel_multi_index.rst "Download source file")
-  .pdf

# jax.numpy.ravel_multi_index

## Contents

- [`ravel_multi_index()`](#jax.numpy.ravel_multi_index)

# jax.numpy.ravel_multi_index[\#](#jax-numpy-ravel-multi-index "Link to this heading")

jax.numpy.ravel_multi_index(*multi_index*, *dims*, *mode='raise'*, *order='C'*, *\**, *dtype=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L2040-L2148)[\#](#jax.numpy.ravel_multi_index "Link to this definition")  
Convert multi-dimensional indices into flat indices.

JAX implementation of [`numpy.ravel_multi_index()`](https://numpy.org/doc/stable/reference/generated/numpy.ravel_multi_index.html#numpy.ravel_multi_index "(in NumPy v2.4)")

Parameters:  
- **multi_index** (*Sequence\[ArrayLike\]*) – sequence of integer arrays containing indices in each dimension.

- **dims** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – sequence of integer sizes; must have `len(dims)`` ``==`` ``len(multi_index)`

- **mode** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) –

  how to handle out-of bound indices. Options are

  - `"raise"` (default): raise a ValueError. This mode is incompatible with [`jit()`](jax.jit.html#jax.jit "jax.jit") or other JAX transformations.

  - `"clip"`: clip out-of-bound indices to valid range.

  - `"wrap"`: wrap out-of-bound indices to valid range.

  - `"ignore"`: do not coerce or check input indices. Behavior is undefined if indices are out of bounds.

- **order** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – `"C"` (default) or `"F"`, specify whether to assume C-style row-major order or Fortran-style column-major order.

- **dtype** (*DTypeLike* *\|* *None*) – the desired output dtype. If not specified, the dtype is determined by standard type promotion rules of the input multi_index.

Returns:  
array of flattened indices

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

[`jax.numpy.unravel_index()`](jax.numpy.unravel_index.html#jax.numpy.unravel_index "jax.numpy.unravel_index"): inverse of this function.

Examples

Define a 2-dimensional array and a sequence of indices of even values:

    >>> x = jnp.array([[2., 3., 4.],
    ...                [5., 6., 7.]])
    >>> indices = jnp.where(x % 2 == 0)
    >>> indices
    (Array([0, 0, 1], dtype=int32), Array([0, 2, 1], dtype=int32))
    >>> x[indices]
    Array([2., 4., 6.], dtype=float32)

Compute the flattened indices:

    >>> indices_flat = jnp.ravel_multi_index(indices, x.shape)
    >>> indices_flat
    Array([0, 2, 4], dtype=int32)

These flattened indices can be used to extract the same values from the flattened `x` array:

    >>> x_flat = x.ravel()
    >>> x_flat
    Array([2., 3., 4., 5., 6., 7.], dtype=float32)
    >>> x_flat[indices_flat]
    Array([2., 4., 6.], dtype=float32)

The original indices can be recovered with [`unravel_index()`](jax.numpy.unravel_index.html#jax.numpy.unravel_index "jax.numpy.unravel_index"):

    >>> jnp.unravel_index(indices_flat, x.shape)
    (Array([0, 0, 1], dtype=int32), Array([0, 2, 1], dtype=int32))

[](jax.numpy.ravel.html "previous page")

previous

jax.numpy.ravel

[](jax.numpy.real.html "next page")

next

jax.numpy.real

Contents

- [`ravel_multi_index()`](#jax.numpy.ravel_multi_index)

By The JAX authors

© Copyright 2024, The JAX Authors.\
