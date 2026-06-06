- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.sparse` module](../jax.experimental.sparse.html)
- jax.experimental.sparse.bcoo_dynamic_slice

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.sparse.bcoo_dynamic_slice.rst "Download source file")
-  .pdf

# jax.experimental.sparse.bcoo_dynamic_slice

## Contents

- [`bcoo_dynamic_slice()`](#jax.experimental.sparse.bcoo_dynamic_slice)

# jax.experimental.sparse.bcoo_dynamic_slice[\#](#jax-experimental-sparse-bcoo-dynamic-slice "Link to this heading")

jax.experimental.sparse.bcoo_dynamic_slice(*mat*, *start_indices*, *slice_sizes*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/experimental/sparse/bcoo.py#L2036-L2116)[\#](#jax.experimental.sparse.bcoo_dynamic_slice "Link to this definition")  
Sparse implementation of [`jax.lax.dynamic_slice()`](jax.lax.dynamic_slice.html#jax.lax.dynamic_slice "jax.lax.dynamic_slice").

Parameters:  
- **mat** ([*BCOO*](jax.experimental.sparse.BCOO.html#jax.experimental.sparse.BCOO "jax.experimental.sparse.BCOO")) – BCOO array to slice.

- **start_indices** (*Sequence\[Any\]*) – a list of scalar indices, one per dimension. These values may be dynamic.

- **slice_sizes** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – the size of the slice. Must be a sequence of non-negative integers with length equal to ndim(operand). Inside a JIT compiled function, only static values are supported (all JAX arrays inside JIT must have statically known size).

Returns:  
BCOO array containing the slice.

Return type:  
out

[](jax.experimental.sparse.bcoo_dot_general_sampled.html "previous page")

previous

jax.experimental.sparse.bcoo_dot_general_sampled

[](jax.experimental.sparse.bcoo_extract.html "next page")

next

jax.experimental.sparse.bcoo_extract

Contents

- [`bcoo_dynamic_slice()`](#jax.experimental.sparse.bcoo_dynamic_slice)

By The JAX authors

© Copyright 2024, The JAX Authors.\
