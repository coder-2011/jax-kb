- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.sparse` module](../jax.experimental.sparse.html)
- jax.experimental.sparse.bcoo_dot_general_sampled

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.sparse.bcoo_dot_general_sampled.rst "Download source file")
-  .pdf

# jax.experimental.sparse.bcoo_dot_general_sampled

## Contents

- [`bcoo_dot_general_sampled()`](#jax.experimental.sparse.bcoo_dot_general_sampled)

# jax.experimental.sparse.bcoo_dot_general_sampled[\#](#jax-experimental-sparse-bcoo-dot-general-sampled "Link to this heading")

jax.experimental.sparse.bcoo_dot_general_sampled(*A*, *B*, *indices*, *\**, *dimension_numbers*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/experimental/sparse/bcoo.py#L943-L964)[\#](#jax.experimental.sparse.bcoo_dot_general_sampled "Link to this definition")  
A contraction operation with output computed at given sparse indices.

Parameters:  
- **lhs** – An ndarray.

- **rhs** – An ndarray.

- **indices** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – BCOO indices.

- **dimension_numbers** (*DotDimensionNumbers*) – a tuple of tuples of the form ((lhs_contracting_dims, rhs_contracting_dims), (lhs_batch_dims, rhs_batch_dims)).

- **A** ([*Array*](jax.Array.html#jax.Array "jax.Array"))

- **B** ([*Array*](jax.Array.html#jax.Array "jax.Array"))

Returns:  
BCOO data, an ndarray containing the result.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.experimental.sparse.bcoo_dot_general.html "previous page")

previous

jax.experimental.sparse.bcoo_dot_general

[](jax.experimental.sparse.bcoo_dynamic_slice.html "next page")

next

jax.experimental.sparse.bcoo_dynamic_slice

Contents

- [`bcoo_dot_general_sampled()`](#jax.experimental.sparse.bcoo_dot_general_sampled)

By The JAX authors

© Copyright 2024, The JAX Authors.\
