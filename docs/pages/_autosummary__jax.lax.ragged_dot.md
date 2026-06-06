- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.ragged_dot

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.ragged_dot.rst "Download source file")
-  .pdf

# jax.lax.ragged_dot

## Contents

- [`ragged_dot()`](#jax.lax.ragged_dot)

# jax.lax.ragged_dot[\#](#jax-lax-ragged-dot "Link to this heading")

jax.lax.ragged_dot(*lhs*, *rhs*, *group_sizes*, *precision=None*, *preferred_element_type=None*, *group_offset=None*, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L2624-L2656)[\#](#jax.lax.ragged_dot "Link to this definition")  
Ragged matrix multiplication.

Parameters:  
- **lhs** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – (m, k) shaped array.

- **rhs** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – (g, k, n) shaped array.

- **group_sizes** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – (g,) shaped array with integer element type, where g denotes number of groups. The ith element indicates the size of ith group.

- **precision** (*PrecisionLike*) – Optional. Consistent with precision argument for [`jax.lax.dot()`](jax.lax.dot.html#jax.lax.dot "jax.lax.dot").

- **preferred_element_type** (*DTypeLike* *\|* *None*) – Optional. Consistent with precision argument for [`jax.lax.dot()`](jax.lax.dot.html#jax.lax.dot "jax.lax.dot").

- **group_offset** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* *None*) – Optional. (1,) shaped array that indicates the group in group_sizes to start computing from. If not specified, defaults to \[0\].

- **out_sharding** ([*NamedSharding*](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding") *\|* *P* *\|* *None*)

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Results:  
(m, n) shaped array with preferred_element_type element type.

[](jax.lax.ragged_all_to_all.html "previous page")

previous

jax.lax.ragged_all_to_all

[](jax.lax.ragged_dot_general.html "next page")

next

jax.lax.ragged_dot_general

Contents

- [`ragged_dot()`](#jax.lax.ragged_dot)

By The JAX authors

© Copyright 2024, The JAX Authors.\
