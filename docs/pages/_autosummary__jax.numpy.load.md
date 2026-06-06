- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.load

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.load.rst "Download source file")
-  .pdf

# jax.numpy.load

## Contents

- [`load()`](#jax.numpy.load)

# jax.numpy.load[\#](#jax-numpy-load "Link to this heading")

jax.numpy.load(*file*, *\*args*, *\*\*kwargs*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L170-L217)[\#](#jax.numpy.load "Link to this definition")  
Load JAX arrays from npy files.

JAX wrapper of [`numpy.load()`](https://numpy.org/doc/stable/reference/generated/numpy.load.html#numpy.load "(in NumPy v2.4)").

This function is a simple wrapper of [`numpy.load()`](https://numpy.org/doc/stable/reference/generated/numpy.load.html#numpy.load "(in NumPy v2.4)"), but in the case of `.npy` files created with [`numpy.save()`](https://numpy.org/doc/stable/reference/generated/numpy.save.html#numpy.save "(in NumPy v2.4)") or [`jax.numpy.save()`](jax.numpy.save.html#jax.numpy.save "jax.numpy.save"), the output will be returned as a [`jax.Array`](jax.Array.html#jax.Array "jax.Array"), and `bfloat16` data types will be restored. For `.npz` files, results will be returned as normal NumPy arrays.

This function requires concrete array inputs, and is not compatible with transformations like [`jax.jit()`](jax.jit.html#jax.jit "jax.jit") or [`jax.vmap()`](jax.vmap.html#jax.vmap "jax.vmap").

Parameters:  
- **file** (*IO\[*[*bytes*](https://docs.python.org/3/library/stdtypes.html#bytes "(in Python v3.14)")*\]* *\|* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* [*os.PathLike*](https://docs.python.org/3/library/os.html#os.PathLike "(in Python v3.14)")*\[Any\]*) – string, bytes, or path-like object containing the array data.

- **args** (*Any*) – for additional arguments, see [`numpy.load()`](https://numpy.org/doc/stable/reference/generated/numpy.load.html#numpy.load "(in NumPy v2.4)")

- **kwargs** (*Any*) – for additional arguments, see [`numpy.load()`](https://numpy.org/doc/stable/reference/generated/numpy.load.html#numpy.load "(in NumPy v2.4)")

Returns:  
the array stored in the file.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.save()`](jax.numpy.save.html#jax.numpy.save "jax.numpy.save"): save an array to a file.

Examples

    >>> import io
    >>> f = io.BytesIO()  # use an in-memory file-like object.
    >>> x = jnp.array([2, 4, 6, 8], dtype='bfloat16')
    >>> jnp.save(f, x)
    >>> f.seek(0)
    0
    >>> jnp.load(f)
    Array([2, 4, 6, 8], dtype=bfloat16)

[](jax.numpy.linspace.html "previous page")

previous

jax.numpy.linspace

[](jax.numpy.log.html "next page")

next

jax.numpy.log

Contents

- [`load()`](#jax.numpy.load)

By The JAX authors

© Copyright 2024, The JAX Authors.\
