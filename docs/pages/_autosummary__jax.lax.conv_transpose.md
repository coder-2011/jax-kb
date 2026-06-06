- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.conv_transpose

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.conv_transpose.rst "Download source file")
-  .pdf

# jax.lax.conv_transpose

## Contents

- [`conv_transpose()`](#jax.lax.conv_transpose)

# jax.lax.conv_transpose[\#](#jax-lax-conv-transpose "Link to this heading")

jax.lax.conv_transpose(*lhs*, *rhs*, *strides*, *padding*, *rhs_dilation=None*, *dimension_numbers=None*, *transpose_kernel=False*, *precision=None*, *preferred_element_type=None*, *use_consistent_padding=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/convolution.py#L296-L389)[\#](#jax.lax.conv_transpose "Link to this definition")  
Convenience wrapper for calculating the N-d convolution “transpose”.

This function directly calculates a fractionally strided conv rather than indirectly calculating the gradient (transpose) of a forward convolution.

Notes

TensorFlow/Keras Compatibility: By default, JAX does NOT reverse the kernel’s spatial dimensions. This differs from TensorFlow’s “Conv2DTranspose” and similar frameworks, which flip spatial axes and swap input/output channels.

To match TensorFlow/Keras behavior, set “transpose_kernel=True” .

Parameters:  
- **lhs** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – a rank n+2 dimensional input array.

- **rhs** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – a rank n+2 dimensional array of kernel weights.

- **strides** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – sequence of n integers, sets fractional stride.

- **padding** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *Sequence\[*[*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]\]*) – ‘SAME’, ‘VALID’, or a sequence of n integer 2-tuples describing before-and-after padding for each spatial dimension. If use_consistent_padding=True, this is interpreted as the padding of the corresponding forward conv, which effectively adds dilation \* (kernel_size - 1) - padding zero padding to each side of the input so that conv_transpose becomes the gradient of conv when given the same padding and stride arguments. This is the behavior in PyTorch. If use_consistent_padding=False, the ‘SAME’ and ‘VALID’ strings are interpreted as the padding of the corresponding forward conv, but integer tuples are interpreted as padding for the transposed convolution.

- **rhs_dilation** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]* *\|* *None*) – None, or a sequence of n integers, giving the dilation factor to apply in each spatial dimension of rhs. RHS dilation is also known as atrous convolution.

- **dimension_numbers** (*ConvGeneralDilatedDimensionNumbers*) – tuple of dimension descriptors as in lax.conv_general_dilated. Defaults to tensorflow convention.

- **transpose_kernel** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – if True flips spatial axes and swaps the input/output channel axes of the kernel. This makes the output of this function identical to the gradient-derived functions like keras.layers.Conv2DTranspose applied to the same kernel. For typical use in neural nets this is completely pointless and just makes input/output channel specification confusing.

- **precision** (*lax.PrecisionLike*) – Optional. Either `None`, which means the default precision for the backend, a [`Precision`](../jax.lax.html#jax.lax.Precision "jax.lax.Precision") enum value (`Precision.DEFAULT`, `Precision.HIGH` or `Precision.HIGHEST`) or a tuple of two [`Precision`](../jax.lax.html#jax.lax.Precision "jax.lax.Precision") enums indicating precision of `` lhs` `` and `rhs`.

- **preferred_element_type** (*DTypeLike* *\|* *None*) – Optional. Either `None`, which means the default accumulation type for the input types, or a datatype, indicating to accumulate results to and return a result with that datatype.

- **use_consistent_padding** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – In older versions of jax, the padding argument was interpreted differently depending on whether it was a string or a sequence of integers. Strings were interpreted as padding for the forward convolution, while integers were interpreted as padding for the transposed convolution. If use_consistent_padding is False, this inconsistent behavior is preserved for backwards compatibility.

Returns:  
Transposed N-d convolution, with output padding following the conventions of keras.layers.Conv2DTranspose.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.lax.conv_general_dilated_patches.html "previous page")

previous

jax.lax.conv_general_dilated_patches

[](jax.lax.conv_with_general_padding.html "next page")

next

jax.lax.conv_with_general_padding

Contents

- [`conv_transpose()`](#jax.lax.conv_transpose)

By The JAX authors

© Copyright 2024, The JAX Authors.\
