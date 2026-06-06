- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.bitcast_convert_type

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.bitcast_convert_type.rst "Download source file")
-  .pdf

# jax.lax.bitcast_convert_type

## Contents

- [`bitcast_convert_type()`](#jax.lax.bitcast_convert_type)

# jax.lax.bitcast_convert_type[\#](#jax-lax-bitcast-convert-type "Link to this heading")

jax.lax.bitcast_convert_type(*operand*, *new_dtype*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L1759-L1793)[\#](#jax.lax.bitcast_convert_type "Link to this definition")  
Elementwise bitcast.

This function lowers directly to the [stablehlo.bitcast_convert](https://openxla.org/stablehlo/spec#bitcast_convert) operation.

The output shape depends on the size of the input and output dtypes with the following logic:

    if new_dtype.itemsize == operand.dtype.itemsize:
      output_shape = operand.shape
    if new_dtype.itemsize < operand.dtype.itemsize:
      output_shape = (*operand.shape, operand.dtype.itemsize // new_dtype.itemsize)
    if new_dtype.itemsize > operand.dtype.itemsize:
      assert operand.shape[-1] * operand.dtype.itemsize == new_dtype.itemsize
      output_shape = operand.shape[:-1]

Parameters:  
- **operand** (*ArrayLike*) – an array or scalar value to be cast

- **new_dtype** (*DTypeLike*) – the new type. Should be a NumPy type.

Returns:  
An array of shape output_shape (see above) and type new_dtype, constructed from the same bits as operand.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.lax.convert_element_type()`](jax.lax.convert_element_type.html#jax.lax.convert_element_type "jax.lax.convert_element_type"): value-preserving dtype conversion.

- [`jax.Array.view()`](jax.Array.view.html#jax.Array.view "jax.Array.view"): NumPy-style API for bitcast type conversion.

[](jax.lax.betainc.html "previous page")

previous

jax.lax.betainc

[](jax.lax.bitwise_and.html "next page")

next

jax.lax.bitwise_and

Contents

- [`bitcast_convert_type()`](#jax.lax.bitcast_convert_type)

By The JAX authors

© Copyright 2024, The JAX Authors.\
