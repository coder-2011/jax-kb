- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.diag

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.diag.rst "Download source file")
-  .pdf

# jax.numpy.diag

## Contents

- [`diag()`](#jax.numpy.diag)

# jax.numpy.diag[\#](#jax-numpy-diag "Link to this heading")

jax.numpy.diag(*v*, *k=0*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L7314-L7364)[\#](#jax.numpy.diag "Link to this definition")  
Returns the specified diagonal or constructs a diagonal array.

JAX implementation of [`numpy.diag()`](https://numpy.org/doc/stable/reference/generated/numpy.diag.html#numpy.diag "(in NumPy v2.4)").

The JAX version always returns a copy of the input, although if this is used within a JIT compilation, the compiler may avoid the copy.

Parameters:  
- **v** (*ArrayLike*) – Input array. Can be a 1-D array to create a diagonal matrix or a 2-D array to extract a diagonal.

- **k** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – optional, default=0. Diagonal offset. Positive values place the diagonal above the main diagonal, negative values place it below the main diagonal.

Returns:  
If v is a 2-D array, a 1-D array containing the diagonal elements. If v is a 1-D array, a 2-D array with the input elements placed along the specified diagonal.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.diagflat()`](jax.numpy.diagflat.html#jax.numpy.diagflat "jax.numpy.diagflat")

- [`jax.numpy.diagonal()`](jax.numpy.diagonal.html#jax.numpy.diagonal "jax.numpy.diagonal")

Examples

Creating a diagonal matrix from a 1-D array:

    >>> jnp.diag(jnp.array([1, 2, 3]))
    Array([[1, 0, 0],
           [0, 2, 0],
           [0, 0, 3]], dtype=int32)

Specifying a diagonal offset:

    >>> jnp.diag(jnp.array([1, 2, 3]), k=1)
    Array([[0, 1, 0, 0],
           [0, 0, 2, 0],
           [0, 0, 0, 3],
           [0, 0, 0, 0]], dtype=int32)

Extracting a diagonal from a 2-D array:

    >>> x = jnp.array([[1, 2, 3],
    ...                [4, 5, 6],
    ...                [7, 8, 9]])
    >>> jnp.diag(x)
    Array([1, 5, 9], dtype=int32)

[](jax.numpy.delete.html "previous page")

previous

jax.numpy.delete

[](jax.numpy.diag_indices.html "next page")

next

jax.numpy.diag_indices

Contents

- [`diag()`](#jax.numpy.diag)

By The JAX authors

© Copyright 2024, The JAX Authors.\
