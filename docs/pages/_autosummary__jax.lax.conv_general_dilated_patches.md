- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.conv_general_dilated_patches

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.conv_general_dilated_patches.rst "Download source file")
-  .pdf

# jax.lax.conv_general_dilated_patches

## Contents

- [`conv_general_dilated_patches()`](#jax.lax.conv_general_dilated_patches)

# jax.lax.conv_general_dilated_patches[\#](#jax-lax-conv-general-dilated-patches "Link to this heading")

jax.lax.conv_general_dilated_patches(*lhs*, *filter_shape*, *window_strides*, *padding*, *lhs_dilation=None*, *rhs_dilation=None*, *dimension_numbers=None*, *precision=None*, *preferred_element_type=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/other.py#L31-L124)[\#](#jax.lax.conv_general_dilated_patches "Link to this definition")  
Extract patches subject to the receptive field of conv_general_dilated.

Runs the input through a convolution with given parameters. The kernel of the convolution is constructed such that the output channel dimension “C” contains flattened image patches, so instead a single “C” dimension represents, for example, three dimensions “chw” collapsed. The order of these dimensions is “c” + ‘’.join(c for c in rhs_spec if c not in ‘OI’), where rhs_spec == dimension_numbers\[1\], and the size of this “C” dimension is therefore the size of each patch, i.e. np.prod(filter_shape) \* lhs.shape\[lhs_spec.index(‘C’)\], where lhs_spec == dimension_numbers\[0\].

Docstring below adapted from jax.lax.conv_general_dilated.

See also

[https://www.openxla.org/xla/operation_semantics#conv_convolution](https://www.openxla.org/xla/operation_semantics#conv_convolution)

Parameters:  
- **lhs** (*ArrayLike*) – a rank n+2 dimensional input array.

- **filter_shape** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – a sequence of n integers, representing the receptive window spatial shape in the order as specified in rhs_spec = dimension_numbers\[1\].

- **window_strides** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – a sequence of n integers, representing the inter-window strides.

- **padding** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *Sequence\[*[*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]\]*) – either the string ‘SAME’, the string ‘VALID’, or a sequence of n (low, high) integer pairs that give the padding to apply before and after each spatial dimension.

- **lhs_dilation** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]* *\|* *None*) – None, or a sequence of n integers, giving the dilation factor to apply in each spatial dimension of lhs. LHS dilation is also known as transposed convolution.

- **rhs_dilation** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]* *\|* *None*) – None, or a sequence of n integers, giving the dilation factor to apply in each spatial dimension of rhs. RHS dilation is also known as atrous convolution.

- **dimension_numbers** (*convolution.ConvGeneralDilatedDimensionNumbers* *\|* *None*) – either None, or a 3-tuple (lhs_spec, rhs_spec, out_spec), where each element is a string of length n+2. None defaults to (“NCHWD…, OIHWD…, NCHWD…”).

- **precision** ([*lax.Precision*](../jax.lax.html#jax.lax.Precision "jax.lax.Precision") *\|* *None*) – Optional. Either `None`, which means the default precision for the backend, or a [`Precision`](../jax.lax.html#jax.lax.Precision "jax.lax.Precision") enum value (`Precision.DEFAULT`, `Precision.HIGH` or `Precision.HIGHEST`).

- **preferred_element_type** (*DType* *\|* *None*) – Optional. Either `None`, which means the default accumulation type for the input types, or a datatype, indicating to accumulate results to and return a result with that datatype.

Returns:  
A rank n+2 array containing the flattened image patches in the output channel (“C”) dimension. For example if dimension_numbers = (“NcHW”, “OIwh”, “CNHW”), the output has dimension numbers “CNHW” = “{cwh}NHW”, with the size of dimension “C” equal to the size of each patch (np.prod(filter_shape) \* lhs.shape\[lhs_spec.index(‘C’)\]).

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.lax.conv_general_dilated_local.html "previous page")

previous

jax.lax.conv_general_dilated_local

[](jax.lax.conv_transpose.html "next page")

next

jax.lax.conv_transpose

Contents

- [`conv_general_dilated_patches()`](#jax.lax.conv_general_dilated_patches)

By The JAX authors

© Copyright 2024, The JAX Authors.\
