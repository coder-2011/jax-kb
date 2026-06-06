- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.issubdtype

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.issubdtype.rst "Download source file")
-  .pdf

# jax.numpy.issubdtype

## Contents

- [`issubdtype()`](#jax.numpy.issubdtype)

# jax.numpy.issubdtype[\#](#jax-numpy-issubdtype "Link to this heading")

jax.numpy.issubdtype(*arg1*, *arg2*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L326-L371)[\#](#jax.numpy.issubdtype "Link to this definition")  
Return True if arg1 is equal or lower than arg2 in the type hierarchy.

JAX implementation of [`numpy.issubdtype()`](https://numpy.org/doc/stable/reference/generated/numpy.issubdtype.html#numpy.issubdtype "(in NumPy v2.4)").

The main difference in JAX’s implementation is that it properly handles dtype extensions such as `bfloat16`.

Parameters:  
- **arg1** (*DTypeLike*) – dtype-like object. In typical usage, this will be a dtype specifier, such as `"float32"` (i.e. a string), `np.dtype('int32')` (i.e. an instance of [`numpy.dtype`](jax.numpy.dtype.html#jax.numpy.dtype "numpy.dtype")), `jnp.complex64` (i.e. a JAX scalar constructor), or `np.uint8` (i.e. a NumPy scalar type).

- **arg2** (*DTypeLike*) – dtype-like object. In typical usage, this will be a generic scalar type, such as `jnp.integer`, `jnp.floating`, or `jnp.complexfloating`.

Returns:  
True if arg1 represents a dtype that is equal or lower in the type hierarchy than arg2.

Return type:  
[bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")

See also

- [`jax.numpy.isdtype()`](jax.numpy.isdtype.html#jax.numpy.isdtype "jax.numpy.isdtype"): similar function aligning with the array API standard.

Examples

    >>> jnp.issubdtype('uint32', jnp.unsignedinteger)
    True
    >>> jnp.issubdtype(np.int32, jnp.integer)
    True
    >>> jnp.issubdtype(jnp.bfloat16, jnp.floating)
    True
    >>> jnp.issubdtype(np.dtype('complex64'), jnp.complexfloating)
    True
    >>> jnp.issubdtype('complex64', jnp.integer)
    False

Be aware that while this is very similar to [`numpy.issubdtype()`](https://numpy.org/doc/stable/reference/generated/numpy.issubdtype.html#numpy.issubdtype "(in NumPy v2.4)"), the results of these differ in the case of JAX’s custom floating point types:

    >>> np.issubdtype('bfloat16', np.floating)
    False
    >>> jnp.issubdtype('bfloat16', jnp.floating)
    True

[](jax.numpy.isscalar.html "previous page")

previous

jax.numpy.isscalar

[](jax.numpy.iterable.html "next page")

next

jax.numpy.iterable

Contents

- [`issubdtype()`](#jax.numpy.issubdtype)

By The JAX authors

© Copyright 2024, The JAX Authors.\
