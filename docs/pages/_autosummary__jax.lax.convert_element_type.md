- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.convert_element_type

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.convert_element_type.rst "Download source file")
-  .pdf

# jax.lax.convert_element_type

## Contents

- [`convert_element_type()`](#jax.lax.convert_element_type)

# jax.lax.convert_element_type[\#](#jax-lax-convert-element-type "Link to this heading")

jax.lax.convert_element_type(*operand*, *new_dtype*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L1628-L1664)[\#](#jax.lax.convert_element_type "Link to this definition")  
Elementwise cast.

This function lowers directly to the [stablehlo.convert](https://openxla.org/stablehlo/spec#convert) operation, which performs an elementwise conversion from one type to another, similar to a C++ `static_cast`.

Parameters:  
- **operand** (*ArrayLike*) – an array or scalar value to be cast.

- **new_dtype** (*DTypeLike* *\|* *dtypes.ExtendedDType*) – a dtype-like object (e.g. a [`numpy.dtype`](jax.numpy.dtype.html#jax.numpy.dtype "numpy.dtype"), a scalar type, or a valid dtype name) representing the target dtype.

Returns:  
An array with the same shape as `operand`, cast elementwise to `new_dtype`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Note

If `new_dtype` is a 64-bit type and [x64 mode](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision) is not enabled, the appropriate 32-bit type will be used in its place.

If the input is a JAX array and the input dtype and output dtype match, then the input array will be returned unmodified.

See also

- [`jax.numpy.astype()`](jax.numpy.astype.html#jax.numpy.astype "jax.numpy.astype"): NumPy-style dtype casting API.

- [`jax.Array.astype()`](jax.Array.astype.html#jax.Array.astype "jax.Array.astype"): dtype casting as an array method.

- [`jax.lax.bitcast_convert_type()`](jax.lax.bitcast_convert_type.html#jax.lax.bitcast_convert_type "jax.lax.bitcast_convert_type"): cast bits directly to a new dtype.

[](jax.lax.conv.html "previous page")

previous

jax.lax.conv

[](jax.lax.conv_dimension_numbers.html "next page")

next

jax.lax.conv_dimension_numbers

Contents

- [`convert_element_type()`](#jax.lax.convert_element_type)

By The JAX authors

© Copyright 2024, The JAX Authors.\
