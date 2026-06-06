- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.conv_general_dilated_local

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.conv_general_dilated_local.rst "Download source file")
-  .pdf

# jax.lax.conv_general_dilated_local

## Contents

- [`conv_general_dilated_local()`](#jax.lax.conv_general_dilated_local)

# jax.lax.conv_general_dilated_local[\#](#jax-lax-conv-general-dilated-local "Link to this heading")

jax.lax.conv_general_dilated_local(*lhs*, *rhs*, *window_strides*, *padding*, *filter_shape*, *lhs_dilation=None*, *rhs_dilation=None*, *dimension_numbers=None*, *precision=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/other.py#L126-L243)[\#](#jax.lax.conv_general_dilated_local "Link to this definition")  
General n-dimensional unshared convolution operator with optional dilation.

Also known as locally connected layer, the operation is equivalent to convolution with a separate (unshared) rhs kernel used at each output spatial location. Docstring below adapted from jax.lax.conv_general_dilated.

See also

[https://www.openxla.org/xla/operation_semantics#conv_convolution](https://www.openxla.org/xla/operation_semantics#conv_convolution)

Parameters:  
- **lhs** (*ArrayLike*) – a rank n+2 dimensional input array.

- **rhs** (*ArrayLike*) – a rank n+2 dimensional array of kernel weights. Unlike in regular CNNs, its spatial coordinates (H, W, …) correspond to output spatial locations, while input spatial locations are fused with the input channel locations in the single I dimension, in the order of “C” + ‘’.join(c for c in rhs_spec if c not in ‘OI’), where rhs_spec = dimension_numbers\[1\]. For example, if rhs_spec == “WHIO”, the unfolded kernel shape is \`”\[output W\]\[output H\]{I\[receptive window W\]\[receptive window H\]}O”.

- **window_strides** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – a sequence of n integers, representing the inter-window strides.

- **padding** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *Sequence\[*[*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]\]*) – either the string ‘SAME’, the string ‘VALID’, or a sequence of n (low, high) integer pairs that give the padding to apply before and after each spatial dimension.

- **filter_shape** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – a sequence of n integers, representing the receptive window spatial shape in the order as specified in rhs_spec = dimension_numbers\[1\].

- **lhs_dilation** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]* *\|* *None*) – None, or a sequence of n integers, giving the dilation factor to apply in each spatial dimension of lhs. LHS dilation is also known as transposed convolution.

- **rhs_dilation** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]* *\|* *None*) – None, or a sequence of n integers, giving the dilation factor to apply in each input spatial dimension of rhs. RHS dilation is also known as atrous convolution.

- **dimension_numbers** (*convolution.ConvGeneralDilatedDimensionNumbers* *\|* *None*) – either None, a ConvDimensionNumbers object, or a 3-tuple (lhs_spec, rhs_spec, out_spec), where each element is a string of length n+2.

- **precision** (*lax.PrecisionLike*) – Optional. Either `None`, which means the default precision for the backend, a `lax.Precision` enum value (`Precision.DEFAULT`, `Precision.HIGH` or `Precision.HIGHEST`) or a tuple of two `lax.Precision` enums indicating precision of `` lhs` `` and `rhs`.

Returns:  
An array containing the unshared convolution result.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

In the string case of dimension_numbers, each character identifies by position:

- the batch dimensions in lhs, rhs, and the output with the character ‘N’,

- the feature dimensions in lhs and the output with the character ‘C’,

- the input and output feature dimensions in rhs with the characters ‘I’ and ‘O’ respectively, and

- spatial dimension correspondences between lhs, rhs, and the output using any distinct characters. The examples below use ‘W’ and ‘H’.

For example, to indicate dimension numbers consistent with the conv function with two spatial dimensions, one could use (‘NCHW’, ‘OIHW’, ‘NCHW’). As another example, to indicate dimension numbers consistent with the TensorFlow Conv2D operation, one could use (‘NHWC’, ‘HWIO’, ‘NHWC’). When using the latter form of convolution dimension specification, window strides are associated with spatial dimension character labels according to the order in which the labels appear in the rhs_spec string, so that window_strides\[0\] is matched with the dimension corresponding to the first character appearing in rhs_spec that is not ‘I’ or ‘O’.

If dimension_numbers is None, the default is (‘NCHW’, ‘OIHW’, ‘NCHW’) (for a 2D convolution).

[](jax.lax.conv_general_dilated.html "previous page")

previous

jax.lax.conv_general_dilated

[](jax.lax.conv_general_dilated_patches.html "next page")

next

jax.lax.conv_general_dilated_patches

Contents

- [`conv_general_dilated_local()`](#jax.lax.conv_general_dilated_local)

By The JAX authors

© Copyright 2024, The JAX Authors.\
