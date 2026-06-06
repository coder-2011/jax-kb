- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.delete

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.delete.rst "Download source file")
-  .pdf

# jax.numpy.delete

## Contents

- [`delete()`](#jax.numpy.delete)

# jax.numpy.delete[\#](#jax-numpy-delete "Link to this heading")

jax.numpy.delete(*arr*, *obj*, *axis=None*, *\**, *assume_unique_indices=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L7577-L7707)[\#](#jax.numpy.delete "Link to this definition")  
Delete entry or entries from an array.

JAX implementation of [`numpy.delete()`](https://numpy.org/doc/stable/reference/generated/numpy.delete.html#numpy.delete "(in NumPy v2.4)").

Parameters:  
- **arr** (*ArrayLike*) – array from which entries will be deleted.

- **obj** (*ArrayLike* *\|* [*slice*](https://docs.python.org/3/library/functions.html#slice "(in Python v3.14)")) – index, indices, or slice to be deleted.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – axis along which entries will be deleted.

- **assume_unique_indices** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – In case of array-like integer (not boolean) indices, assume the indices are unique, and perform the deletion in a way that is compatible with JIT and other JAX transformations.

Returns:  
Copy of `arr` with specified indices deleted.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Note

`delete()` usually requires the index specification to be static. If the index is an integer array that is guaranteed to contain unique entries, you may specify `assume_unique_indices=True` to perform the operation in a manner that does not require static indices.

See also

- [`jax.numpy.insert()`](jax.numpy.insert.html#jax.numpy.insert "jax.numpy.insert"): insert entries into an array.

Examples

Delete entries from a 1D array:

    >>> a = jnp.array([4, 5, 6, 7, 8, 9])
    >>> jnp.delete(a, 2)
    Array([4, 5, 7, 8, 9], dtype=int32)
    >>> jnp.delete(a, slice(1, 4))  # delete a[1:4]
    Array([4, 8, 9], dtype=int32)
    >>> jnp.delete(a, slice(None, None, 2))  # delete a[::2]
    Array([5, 7, 9], dtype=int32)

Delete entries from a 2D array along a specified axis:

    >>> a2 = jnp.array([[4, 5, 6],
    ...                 [7, 8, 9]])
    >>> jnp.delete(a2, 1, axis=1)
    Array([[4, 6],
           [7, 9]], dtype=int32)

Delete multiple entries via a sequence of indices:

    >>> indices = jnp.array([0, 1, 3])
    >>> jnp.delete(a, indices)
    Array([6, 8, 9], dtype=int32)

This will fail under [`jit()`](jax.jit.html#jax.jit "jax.jit") and other transformations, because the output shape cannot be known with the possibility of duplicate indices:

    >>> jax.jit(jnp.delete)(a, indices)  
    Traceback (most recent call last):
      ...
    ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected: traced array with shape int32[3].

If you can ensure that the indices are unique, pass `assume_unique_indices` to allow this to be executed under JIT:

    >>> jit_delete = jax.jit(jnp.delete, static_argnames=['assume_unique_indices'])
    >>> jit_delete(a, indices, assume_unique_indices=True)
    Array([6, 8, 9], dtype=int32)

[](jax.numpy.degrees.html "previous page")

previous

jax.numpy.degrees

[](jax.numpy.diag.html "next page")

next

jax.numpy.diag

Contents

- [`delete()`](#jax.numpy.delete)

By The JAX authors

© Copyright 2024, The JAX Authors.\
