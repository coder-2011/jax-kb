- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.ragged_dot_general

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.ragged_dot_general.rst "Download source file")
-  .pdf

# jax.lax.ragged_dot_general

## Contents

- [`ragged_dot_general()`](#jax.lax.ragged_dot_general)

# jax.lax.ragged_dot_general[\#](#jax-lax-ragged-dot-general "Link to this heading")

jax.lax.ragged_dot_general(*lhs*, *rhs*, *group_sizes*, *ragged_dot_dimension_numbers*, *precision=None*, *preferred_element_type=None*, *group_offset=None*, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L2704-L2769)[\#](#jax.lax.ragged_dot_general "Link to this definition")  
Ragged matrix multiplication.

Ragged dot takes three arrays—`lhs`, `rhs`, and `group_sizes`—and a `ragged_dot_dimension_numbers` argument. Like dot_general, `lhs` and `rhs` are allowed arbitrary batch and contracting dimensions. Additionally, `lhs` is required to have one ragged dimension, and `rhs` may have at most one group dimension.

Let g be the number of groups in the lhs ragged dimension. Ragged dot has three modes, depending on the kind of the lhs ragged dimension:

1.  `[b...,m...,k...],`` ``[g,b...,k...,n...],`` ``[b...,x...,g]`` ``->`` ``[b...,m...,n...]`. Here the ragged dimension is a non-contracting dimension (`m`) of `lhs`, and `x...` are the lhs non-contracting dims outer to the ragged dim.

2.  `[b...,m...,k...],`` ``[b...,k...,n...],`` ``[b...,x...,g]`` ``->`` ``[g,b...,m...,n...]`. Here the ragged dimension is a contracting dimension (`k`) of `lhs` and `rhs`, and x… are the lhs contracting dims outer to the ragged dim.

3.  `[b...,m...,k...],`` ``[b...,k...,n...],`` ``[x...,g]`` ``->`` ``[b...,m...,n...]`. Here the ragged dimension is a batch dimension (`b`) of `lhs` and `rhs`, and `x...` are the lhs batch dims outer to the ragged dim.

If `group_sizes` is passed-in with shape `[g]`, it is broadcasted according to the rules above.

Parameters:  
- **lhs** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – an array

- **rhs** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – an array

- **group_sizes** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – an array with integer element type

- **ragged_dot_dimension_numbers** ([*RaggedDotDimensionNumbers*](../jax.lax.html#jax.lax.RaggedDotDimensionNumbers "jax.lax.RaggedDotDimensionNumbers")) – a `RaggedDotDimensionNumbers` object to specify the dot dimension numbers, lhs ragged dimension, and rhs group dimension.

- **precision** (*PrecisionLike*) – Optional. Consistent with precision argument for [`jax.lax.dot()`](jax.lax.dot.html#jax.lax.dot "jax.lax.dot").

- **preferred_element_type** (*DTypeLike* *\|* *None*) – Optional. Consistent with precision argument for [`jax.lax.dot()`](jax.lax.dot.html#jax.lax.dot "jax.lax.dot").

- **group_offset** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* *None*) – Optional. (1,) shaped array that indicates the group in group_sizes to start computing from. If not specified, defaults to \[0\].

- **out_sharding** ([*NamedSharding*](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding") *\|* *P* *\|* *None*)

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Results:  
An array whose shape is the same as that produced by dot_general, with an extra leading dimension of size g in the case where the lhs ragged dimension is a contracting dimension.

[](jax.lax.ragged_dot.html "previous page")

previous

jax.lax.ragged_dot

[](jax.lax.real.html "next page")

next

jax.lax.real

Contents

- [`ragged_dot_general()`](#jax.lax.ragged_dot_general)

By The JAX authors

© Copyright 2024, The JAX Authors.\
