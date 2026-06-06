- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.conv_dimension_numbers

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.conv_dimension_numbers.rst "Download source file")
-  .pdf

# jax.lax.conv_dimension_numbers

## Contents

- [`conv_dimension_numbers()`](#jax.lax.conv_dimension_numbers)

# jax.lax.conv_dimension_numbers[\#](#jax-lax-conv-dimension-numbers "Link to this heading")

jax.lax.conv_dimension_numbers(*lhs_shape*, *rhs_shape*, *dimension_numbers*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/convolution.py#L948-L989)[\#](#jax.lax.conv_dimension_numbers "Link to this definition")  
Converts convolution dimension_numbers to a ConvDimensionNumbers.

Parameters:  
- **lhs_shape** – tuple of nonnegative integers, shape of the convolution input.

- **rhs_shape** – tuple of nonnegative integers, shape of the convolution kernel.

- **dimension_numbers** – None or a tuple/list of strings or a ConvDimensionNumbers object.

Returns:  
A ConvDimensionNumbers object that represents dimension_numbers in the canonical form used by lax functions.

Return type:  
[ConvDimensionNumbers](../jax.lax.html#jax.lax.ConvDimensionNumbers "jax.lax.ConvDimensionNumbers")

[](jax.lax.convert_element_type.html "previous page")

previous

jax.lax.convert_element_type

[](jax.lax.conv_general_dilated.html "next page")

next

jax.lax.conv_general_dilated

Contents

- [`conv_dimension_numbers()`](#jax.lax.conv_dimension_numbers)

By The JAX authors

© Copyright 2024, The JAX Authors.\
