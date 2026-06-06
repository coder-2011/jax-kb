- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.conv

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.conv.rst "Download source file")
-  .pdf

# jax.lax.conv

## Contents

- [`conv()`](#jax.lax.conv)

# jax.lax.conv[\#](#jax-lax-conv "Link to this heading")

jax.lax.conv(*lhs*, *rhs*, *window_strides*, *padding*, *precision=None*, *preferred_element_type=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/convolution.py#L194-L219)[\#](#jax.lax.conv "Link to this definition")  
Convenience wrapper around conv_general_dilated.

Parameters:  
- **lhs** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – a rank n+2 dimensional input array.

- **rhs** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – a rank n+2 dimensional array of kernel weights.

- **window_strides** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – a sequence of n integers, representing the inter-window strides.

- **padding** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – either the string ‘SAME’, the string ‘VALID’.

- **precision** (*lax.PrecisionLike*) – Optional. Either `None`, which means the default precision for the backend, a [`Precision`](../jax.lax.html#jax.lax.Precision "jax.lax.Precision") enum value (`Precision.DEFAULT`, `Precision.HIGH` or `Precision.HIGHEST`) or a tuple of two [`Precision`](../jax.lax.html#jax.lax.Precision "jax.lax.Precision") enums indicating precision of `` lhs` `` and `rhs`.

- **preferred_element_type** (*DTypeLike* *\|* *None*) – Optional. Either `None`, which means the default accumulation type for the input types, or a datatype, indicating to accumulate results to and return a result with that datatype.

Returns:  
An array containing the convolution result.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.lax.conj.html "previous page")

previous

jax.lax.conj

[](jax.lax.convert_element_type.html "next page")

next

jax.lax.convert_element_type

Contents

- [`conv()`](#jax.lax.conv)

By The JAX authors

© Copyright 2024, The JAX Authors.\
