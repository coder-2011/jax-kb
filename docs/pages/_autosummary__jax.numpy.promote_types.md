- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.promote_types

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.promote_types.rst "Download source file")
-  .pdf

# jax.numpy.promote_types

## Contents

- [`promote_types()`](#jax.numpy.promote_types)

# jax.numpy.promote_types[\#](#jax-numpy-promote-types "Link to this heading")

jax.numpy.promote_types(*a*, *b*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/dtypes.py#L910-L959)[\#](#jax.numpy.promote_types "Link to this definition")  
Returns the type to which a binary operation should cast its arguments.

JAX implementation of [`numpy.promote_types()`](https://numpy.org/doc/stable/reference/generated/numpy.promote_types.html#numpy.promote_types "(in NumPy v2.4)"). For details of JAX’s type promotion semantics, see [Type promotion semantics](../type_promotion.html#type-promotion).

Parameters:  
- **a** (*DTypeLike*) – a [`numpy.dtype`](jax.numpy.dtype.html#jax.numpy.dtype "numpy.dtype") or a dtype specifier.

- **b** (*DTypeLike*) – a [`numpy.dtype`](jax.numpy.dtype.html#jax.numpy.dtype "numpy.dtype") or a dtype specifier.

Returns:  
A [`numpy.dtype`](jax.numpy.dtype.html#jax.numpy.dtype "numpy.dtype") object.

Return type:  
DType

Examples

Type specifiers may be strings, dtypes, or scalar types, and the return value is always a dtype:

    >>> jnp.promote_types('int32', 'float32')  # strings
    dtype('float32')
    >>> jnp.promote_types(jnp.dtype('int32'), jnp.dtype('float32'))  # dtypes
    dtype('float32')
    >>> jnp.promote_types(jnp.int32, jnp.float32)  # scalar types
    dtype('float32')

Built-in scalar types (`int`, `float`, or `complex`) are treated as weakly-typed and will not change the bit width of a strongly-typed counterpart (see discussion in [Type promotion semantics](../type_promotion.html#type-promotion)):

    >>> jnp.promote_types('uint8', int)
    dtype('uint8')
    >>> jnp.promote_types('float16', float)
    dtype('float16')

This differs from the NumPy version of this function, which treats built-in scalar types as equivalent to 64-bit types:

    >>> import numpy
    >>> numpy.promote_types('uint8', int)
    dtype('int64')
    >>> numpy.promote_types('float16', float)
    dtype('float64')

[](jax.numpy.prod.html "previous page")

previous

jax.numpy.prod

[](jax.numpy.ptp.html "next page")

next

jax.numpy.ptp

Contents

- [`promote_types()`](#jax.numpy.promote_types)

By The JAX authors

© Copyright 2024, The JAX Authors.\
