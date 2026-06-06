- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.diagonal

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.diagonal.rst "Download source file")
-  .pdf

# jax.numpy.diagonal

## Contents

- [`diagonal()`](#jax.numpy.diagonal)

# jax.numpy.diagonal[\#](#jax-numpy-diagonal "Link to this heading")

jax.numpy.diagonal(*a*, *offset=0*, *axis1=0*, *axis2=1*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L7239-L7312)[\#](#jax.numpy.diagonal "Link to this definition")  
Returns the specified diagonal of an array.

JAX implementation of [`numpy.diagonal()`](https://numpy.org/doc/stable/reference/generated/numpy.diagonal.html#numpy.diagonal "(in NumPy v2.4)").

The JAX version always returns a copy of the input, although if this is used within a JIT compilation, the compiler may avoid the copy.

Parameters:  
- **a** (*ArrayLike*) – Input array. Must be at least 2-dimensional.

- **offset** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – optional, default=0. Diagonal offset from the main diagonal. Must be a static integer value. Can be positive or negative.

- **axis1** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – optional, default=0. The first axis along which to take the diagonal.

- **axis2** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) –

  optional, default=1. The second axis along which to take the diagonal.

  Returns:  
  A 1D array for 2D input, and in general a N-1 dimensional array for N-dimensional input.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.diag()`](jax.numpy.diag.html#jax.numpy.diag "jax.numpy.diag")

- [`jax.numpy.diagflat()`](jax.numpy.diagflat.html#jax.numpy.diagflat "jax.numpy.diagflat")

Examples

    >>> x = jnp.array([[1, 2, 3],
    ...                [4, 5, 6],
    ...                [7, 8, 9]])
    >>> jnp.diagonal(x)
    Array([1, 5, 9], dtype=int32)
    >>> jnp.diagonal(x, offset=1)
    Array([2, 6], dtype=int32)
    >>> jnp.diagonal(x, offset=-1)
    Array([4, 8], dtype=int32)

[](jax.numpy.diagflat.html "previous page")

previous

jax.numpy.diagflat

[](jax.numpy.diff.html "next page")

next

jax.numpy.diff

Contents

- [`diagonal()`](#jax.numpy.diagonal)

By The JAX authors

© Copyright 2024, The JAX Authors.\
