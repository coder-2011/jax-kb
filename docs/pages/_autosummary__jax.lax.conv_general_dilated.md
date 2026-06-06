- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.conv_general_dilated

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.conv_general_dilated.rst "Download source file")
-  .pdf

# jax.lax.conv_general_dilated

## Contents

- [`conv_general_dilated()`](#jax.lax.conv_general_dilated)

# jax.lax.conv_general_dilated[\#](#jax-lax-conv-general-dilated "Link to this heading")

jax.lax.conv_general_dilated(*lhs*, *rhs*, *window_strides*, *padding*, *lhs_dilation=None*, *rhs_dilation=None*, *dimension_numbers=None*, *feature_group_count=1*, *batch_group_count=1*, *precision=None*, *preferred_element_type=None*, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/convolution.py#L61-L189)[\#](#jax.lax.conv_general_dilated "Link to this definition")  
General n-dimensional convolution operator, with optional dilation.

Wraps XLA’s [Conv](https://www.openxla.org/xla/operation_semantics#conv_convolution) operator.

Parameters:  
- **lhs** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – a rank n+2 dimensional input array.

- **rhs** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – a rank n+2 dimensional array of kernel weights.

- **window_strides** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – a sequence of n integers, representing the inter-window strides.

- **padding** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *Sequence\[*[*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]\]*) – either the strings ‘SAME’, ‘SAME_LOWER’, or ‘VALID’, or a sequence of n (low, high) integer pairs that give the padding to apply before and after each spatial dimension. ‘SAME’ and ‘SAME_LOWER’ add padding to produce same output size as the input. The padding is split between the two sides equally or almost equally. In case the padding is an odd number, the extra padding is added at the end for ‘SAME’ and at the beginning for ‘SAME_LOWER’.

- **lhs_dilation** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]* *\|* *None*) – None, or a sequence of n integers, giving the dilation factor to apply in each spatial dimension of lhs. LHS dilation is also known as transposed convolution.

- **rhs_dilation** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]* *\|* *None*) – None, or a sequence of n integers, giving the dilation factor to apply in each spatial dimension of rhs. RHS dilation is also known as atrous convolution.

- **dimension_numbers** (*ConvGeneralDilatedDimensionNumbers*) – either None, a `ConvDimensionNumbers` object, or a 3-tuple `(lhs_spec,`` ``rhs_spec,`` ``out_spec)`, where each element is a string of length n+2.

- **feature_group_count** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – integer, default 1. See XLA HLO docs.

- **batch_group_count** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – integer, default 1. See XLA HLO docs.

- **precision** (*lax.PrecisionLike*) – Optional. Either `None`, which means the default precision for the backend, a [`Precision`](../jax.lax.html#jax.lax.Precision "jax.lax.Precision") enum value (`Precision.DEFAULT`, `Precision.HIGH` or `Precision.HIGHEST`), a string (e.g. ‘highest’ or ‘fastest’, see the `jax.default_matmul_precision` context manager), or a tuple of two [`Precision`](../jax.lax.html#jax.lax.Precision "jax.lax.Precision") enums or strings indicating precision of `lhs` and `rhs`.

- **preferred_element_type** (*DTypeLike* *\|* *None*) – Optional. Either `None`, which means the default accumulation type for the input types, or a datatype, indicating to accumulate results to and return a result with that datatype.

- **out_sharding** ([*NamedSharding*](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding") *\|* *P* *\|* *None*) – Optional. Specifies how the output array should be sharded across devices in multi-device computation. Can be a [`NamedSharding`](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding"), a [`PartitionSpec`](../jax.sharding.html#jax.sharding.PartitionSpec "jax.sharding.PartitionSpec") (`P`), or `None` (default). When specified, the output will be sharded according to the given sharding specification. Primarily used in explicit sharding mode. See the [explicit sharding tutorial](https://docs.jax.dev/en/latest/parallel.html) for more details.

Returns:  
An array containing the convolution result.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

In the string case of `dimension_numbers`, each character identifies by position:

- the batch dimensions in `lhs`, `rhs`, and the output with the character ‘N’,

- the feature dimensions in lhs and the output with the character ‘C’,

- the input and output feature dimensions in rhs with the characters ‘I’ and ‘O’ respectively, and

- spatial dimension correspondences between lhs, rhs, and the output using any distinct characters. The examples below use ‘W’ and ‘H’.

For example, to indicate dimension numbers consistent with the `conv` function with two spatial dimensions, one could use `('NCHW',`` ``'OIHW',`` ``'NCHW')`. As another example, to indicate dimension numbers consistent with the TensorFlow Conv2D operation, one could use `('NHWC',`` ``'HWIO',`` ``'NHWC')`. When using the latter form of convolution dimension specification, window strides are associated with spatial dimension character labels according to the order in which the labels appear in the `rhs_spec` string, so that `window_strides[0]` is matched with the dimension corresponding to the first character appearing in rhs_spec that is not `'I'` or `'O'`.

If `dimension_numbers` is `None`, the default is `('NCHW',`` ``'OIHW',`` ``'NCHW')` (for a 2D convolution).

[](jax.lax.conv_dimension_numbers.html "previous page")

previous

jax.lax.conv_dimension_numbers

[](jax.lax.conv_general_dilated_local.html "next page")

next

jax.lax.conv_general_dilated_local

Contents

- [`conv_general_dilated()`](#jax.lax.conv_general_dilated)

By The JAX authors

© Copyright 2024, The JAX Authors.\
