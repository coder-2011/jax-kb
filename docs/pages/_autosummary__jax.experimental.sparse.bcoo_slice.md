- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.sparse` module](../jax.experimental.sparse.html)
- jax.experimental.sparse.bcoo_slice

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.sparse.bcoo_slice.rst "Download source file")
-  .pdf

# jax.experimental.sparse.bcoo_slice

## Contents

- [`bcoo_slice()`](#jax.experimental.sparse.bcoo_slice)

# jax.experimental.sparse.bcoo_slice[\#](#jax-experimental-sparse-bcoo-slice "Link to this heading")

jax.experimental.sparse.bcoo_slice(*mat*, *\**, *start_indices*, *limit_indices*, *strides=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/experimental/sparse/bcoo.py#L1959-L2035)[\#](#jax.experimental.sparse.bcoo_slice "Link to this definition")  
Sparse implementation of [`jax.lax.slice()`](jax.lax.slice.html#jax.lax.slice "jax.lax.slice").

Parameters:  
- **mat** ([*BCOO*](jax.experimental.sparse.BCOO.html#jax.experimental.sparse.BCOO "jax.experimental.sparse.BCOO")) – BCOO array to be reshaped.

- **start_indices** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – sequence of integers of length mat.ndim specifying the starting indices of each slice.

- **limit_indices** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – sequence of integers of length mat.ndim specifying the ending indices of each slice

- **strides** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]* *\|* *None*) – (not implemented) sequence of integers of length mat.ndim specifying the stride for each slice

Returns:  
BCOO array containing the slice.

Return type:  
out

[](jax.experimental.sparse.bcoo_reshape.html "previous page")

previous

jax.experimental.sparse.bcoo_reshape

[](jax.experimental.sparse.bcoo_sort_indices.html "next page")

next

jax.experimental.sparse.bcoo_sort_indices

Contents

- [`bcoo_slice()`](#jax.experimental.sparse.bcoo_slice)

By The JAX authors

© Copyright 2024, The JAX Authors.\
