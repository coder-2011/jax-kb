- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.sparse` module](../jax.experimental.sparse.html)
- jax.experimental.sparse.bcoo_dot_general

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.sparse.bcoo_dot_general.rst "Download source file")
-  .pdf

# jax.experimental.sparse.bcoo_dot_general

## Contents

- [`bcoo_dot_general()`](#jax.experimental.sparse.bcoo_dot_general)

# jax.experimental.sparse.bcoo_dot_general[\#](#jax-experimental-sparse-bcoo-dot-general "Link to this heading")

jax.experimental.sparse.bcoo_dot_general(*lhs*, *rhs*, *\**, *dimension_numbers*, *precision=None*, *preferred_element_type=None*, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/experimental/sparse/bcoo.py#L611-L654)[\#](#jax.experimental.sparse.bcoo_dot_general "Link to this definition")  
A general contraction operation.

Parameters:  
- **lhs** ([*BCOO*](jax.experimental.sparse.BCOO.html#jax.experimental.sparse.BCOO "jax.experimental.sparse.BCOO") *\|* [*Array*](jax.Array.html#jax.Array "jax.Array")) – An ndarray or BCOO-format sparse array.

- **rhs** ([*BCOO*](jax.experimental.sparse.BCOO.html#jax.experimental.sparse.BCOO "jax.experimental.sparse.BCOO") *\|* [*Array*](jax.Array.html#jax.Array "jax.Array")) – An ndarray or BCOO-format sparse array..

- **dimension_numbers** (*DotDimensionNumbers*) – a tuple of tuples of the form ((lhs_contracting_dims, rhs_contracting_dims), (lhs_batch_dims, rhs_batch_dims)).

- **precision** (*None*) – unused

- **preferred_element_type** (*None*) – unused

Returns:  
An ndarray or BCOO-format sparse array containing the result. If both inputs are sparse, the result will be sparse, of type BCOO. If either input is dense, the result will be dense, of type ndarray.

Return type:  
[BCOO](jax.experimental.sparse.BCOO.html#jax.experimental.sparse.BCOO "jax.experimental.sparse.BCOO") \| [Array](jax.Array.html#jax.Array "jax.Array")

[](jax.experimental.sparse.bcoo_concatenate.html "previous page")

previous

jax.experimental.sparse.bcoo_concatenate

[](jax.experimental.sparse.bcoo_dot_general_sampled.html "next page")

next

jax.experimental.sparse.bcoo_dot_general_sampled

Contents

- [`bcoo_dot_general()`](#jax.experimental.sparse.bcoo_dot_general)

By The JAX authors

© Copyright 2024, The JAX Authors.\
