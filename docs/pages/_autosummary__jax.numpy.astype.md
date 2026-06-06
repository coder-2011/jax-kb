- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.astype

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.astype.rst "Download source file")
-  .pdf

# jax.numpy.astype

## Contents

- [`astype()`](#jax.numpy.astype)

# jax.numpy.astype[\#](#jax-numpy-astype "Link to this heading")

jax.numpy.astype(*x*, *dtype*, */*, *\**, *copy=False*, *device=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L5257-L5320)[\#](#jax.numpy.astype "Link to this definition")  
Convert an array to a specified dtype.

JAX implementation of [`numpy.astype()`](https://numpy.org/doc/stable/reference/generated/numpy.astype.html#numpy.astype "(in NumPy v2.4)").

This is implemented via [`jax.lax.convert_element_type()`](jax.lax.convert_element_type.html#jax.lax.convert_element_type "jax.lax.convert_element_type"), which may have slightly different behavior than [`numpy.astype()`](https://numpy.org/doc/stable/reference/generated/numpy.astype.html#numpy.astype "(in NumPy v2.4)") in some cases. In particular, the details of float-to-int and int-to-float casts are implementation dependent.

Parameters:  
- **x** (*ArrayLike*) – input array to convert

- **dtype** (*DTypeLike* *\|* *None*) – output dtype

- **copy** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – if True, then always return a copy. If False (default) then only return a copy if necessary.

- **device** (*xc.Device* *\|* [*Sharding*](../jax.sharding.html#jax.sharding.Sharding "jax.sharding.Sharding") *\|* *None*) – optionally specify the device to which the output will be committed.

Returns:  
An array with the same shape as `x`, containing values of the specified dtype.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.lax.convert_element_type()`](jax.lax.convert_element_type.html#jax.lax.convert_element_type "jax.lax.convert_element_type"): lower-level function for XLA-style dtype conversions.

Examples

    >>> x = jnp.array([0, 1, 2, 3])
    >>> x
    Array([0, 1, 2, 3], dtype=int32)
    >>> x.astype('float32')
    Array([0.0, 1.0, 2.0, 3.0], dtype=float32)

    >>> y = jnp.array([0.0, 0.5, 1.0])
    >>> y.astype(int)  # truncates fractional values
    Array([0, 0, 1], dtype=int32)

[](jax.numpy.asinh.html "previous page")

previous

jax.numpy.asinh

[](jax.numpy.atan.html "next page")

next

jax.numpy.atan

Contents

- [`astype()`](#jax.numpy.astype)

By The JAX authors

© Copyright 2024, The JAX Authors.\
