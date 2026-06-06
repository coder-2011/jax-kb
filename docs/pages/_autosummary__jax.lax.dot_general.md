- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.dot_general

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.dot_general.rst "Download source file")
-  .pdf

# jax.lax.dot_general

## Contents

- [`dot_general()`](#jax.lax.dot_general)

# jax.lax.dot_general[\#](#jax-lax-dot-general "Link to this heading")

jax.lax.dot_general(*lhs*, *rhs*, *dimension_numbers*, *precision=None*, *preferred_element_type=None*, *\**, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L2522-L2536)[\#](#jax.lax.dot_general "Link to this definition")  
Alias of [`jax.lax.dot()`](jax.lax.dot.html#jax.lax.dot "jax.lax.dot").

Prefer use of [`jax.lax.dot()`](jax.lax.dot.html#jax.lax.dot "jax.lax.dot") directly, but note that it requires all arguments after `lhs` and `rhs` to be specified by keyword rather than position.

Parameters:  
- **lhs** (*ArrayLike*)

- **rhs** (*ArrayLike*)

- **dimension_numbers** (*DotDimensionNumbers*)

- **precision** (*PrecisionLike*)

- **preferred_element_type** (*DTypeLike* *\|* *None*)

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.lax.dot.html "previous page")

previous

jax.lax.dot

[](jax.lax.dynamic_index_in_dim.html "next page")

next

jax.lax.dynamic_index_in_dim

Contents

- [`dot_general()`](#jax.lax.dot_general)

By The JAX authors

© Copyright 2024, The JAX Authors.\
