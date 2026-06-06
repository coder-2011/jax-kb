- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.conv_with_general_padding

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.conv_with_general_padding.rst "Download source file")
-  .pdf

# jax.lax.conv_with_general_padding

## Contents

- [`conv_with_general_padding()`](#jax.lax.conv_with_general_padding)

# jax.lax.conv_with_general_padding[\#](#jax-lax-conv-with-general-padding "Link to this heading")

jax.lax.conv_with_general_padding(*lhs*, *rhs*, *window_strides*, *padding*, *lhs_dilation*, *rhs_dilation*, *precision=None*, *preferred_element_type=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/convolution.py#L220-L258)[\#](#jax.lax.conv_with_general_padding "Link to this definition")  
Convenience wrapper around conv_general_dilated.

Parameters:  
- **lhs** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – a rank n+2 dimensional input array.

- **rhs** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – a rank n+2 dimensional array of kernel weights.

- **window_strides** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – a sequence of n integers, representing the inter-window strides.

- **padding** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *Sequence\[*[*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]\]*) – either the string ‘SAME’, the string ‘VALID’, or a sequence of n (low, high) integer pairs that give the padding to apply before and after each spatial dimension.

- **lhs_dilation** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]* *\|* *None*) – None, or a sequence of n integers, giving the dilation factor to apply in each spatial dimension of lhs. LHS dilation is also known as transposed convolution.

- **rhs_dilation** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]* *\|* *None*) – None, or a sequence of n integers, giving the dilation factor to apply in each spatial dimension of rhs. RHS dilation is also known as atrous convolution.

- **precision** (*lax.PrecisionLike*) – Optional. Either `None`, which means the default precision for the backend, a [`Precision`](../jax.lax.html#jax.lax.Precision "jax.lax.Precision") enum value (`Precision.DEFAULT`, `Precision.HIGH` or `Precision.HIGHEST`) or a tuple of two [`Precision`](../jax.lax.html#jax.lax.Precision "jax.lax.Precision") enums indicating precision of `` lhs` `` and `rhs`.

- **preferred_element_type** (*DTypeLike* *\|* *None*) – Optional. Either `None`, which means the default accumulation type for the input types, or a datatype, indicating to accumulate results to and return a result with that datatype.

Returns:  
An array containing the convolution result.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.lax.conv_transpose.html "previous page")

previous

jax.lax.conv_transpose

[](jax.lax.cos.html "next page")

next

jax.lax.cos

Contents

- [`conv_with_general_padding()`](#jax.lax.conv_with_general_padding)

By The JAX authors

© Copyright 2024, The JAX Authors.\
