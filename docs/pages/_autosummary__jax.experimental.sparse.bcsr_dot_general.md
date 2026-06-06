- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.sparse` module](../jax.experimental.sparse.html)
- jax.experimental.sparse.bcsr_dot_general

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.sparse.bcsr_dot_general.rst "Download source file")
-  .pdf

# jax.experimental.sparse.bcsr_dot_general

## Contents

- [`bcsr_dot_general()`](#jax.experimental.sparse.bcsr_dot_general)

# jax.experimental.sparse.bcsr_dot_general[\#](#jax-experimental-sparse-bcsr-dot-general "Link to this heading")

jax.experimental.sparse.bcsr_dot_general(*lhs*, *rhs*, *\**, *dimension_numbers*, *precision=None*, *preferred_element_type=None*, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/experimental/sparse/bcsr.py#L464-L500)[\#](#jax.experimental.sparse.bcsr_dot_general "Link to this definition")  
A general contraction operation.

Parameters:  
- **lhs** ([*BCSR*](jax.experimental.sparse.BCSR.html#jax.experimental.sparse.BCSR "jax.experimental.sparse.BCSR") *\|* [*Array*](jax.Array.html#jax.Array "jax.Array")) – An ndarray or BCSR-format sparse array.

- **rhs** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – An ndarray or BCSR-format sparse array..

- **dimension_numbers** (*DotDimensionNumbers*) – a tuple of tuples of the form ((lhs_contracting_dims, rhs_contracting_dims), (lhs_batch_dims, rhs_batch_dims)).

- **precision** (*None*) – unused

- **preferred_element_type** (*None*) – unused

Returns:  
An ndarray or BCSR-format sparse array containing the result. If both inputs are sparse, the result will be sparse, of type BCSR. If either input is dense, the result will be dense, of type ndarray.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.experimental.sparse.BCSR.html "previous page")

previous

jax.experimental.sparse.BCSR

[](jax.experimental.sparse.bcsr_extract.html "next page")

next

jax.experimental.sparse.bcsr_extract

Contents

- [`bcsr_dot_general()`](#jax.experimental.sparse.bcsr_dot_general)

By The JAX authors

© Copyright 2024, The JAX Authors.\
