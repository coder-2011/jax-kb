- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.sparse` module](../jax.experimental.sparse.html)
- jax.experimental.sparse.bcsr_extract

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.sparse.bcsr_extract.rst "Download source file")
-  .pdf

# jax.experimental.sparse.bcsr_extract

## Contents

- [`bcsr_extract()`](#jax.experimental.sparse.bcsr_extract)

# jax.experimental.sparse.bcsr_extract[\#](#jax-experimental-sparse-bcsr-extract "Link to this heading")

jax.experimental.sparse.bcsr_extract(*indices*, *indptr*, *mat*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/experimental/sparse/bcsr.py#L388-L400)[\#](#jax.experimental.sparse.bcsr_extract "Link to this definition")  
Extract values from a dense matrix at given BCSR (indices, indptr).

Parameters:  
- **indices** (*ArrayLike*) – An ndarray; see BCSR indices.

- **indptr** (*ArrayLike*) – An ndarray; see BCSR indptr.

- **mat** (*ArrayLike*) – A dense matrix.

Returns:  
An ndarray; see BCSR data.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.experimental.sparse.bcsr_dot_general.html "previous page")

previous

jax.experimental.sparse.bcsr_dot_general

[](jax.experimental.sparse.bcsr_fromdense.html "next page")

next

jax.experimental.sparse.bcsr_fromdense

Contents

- [`bcsr_extract()`](#jax.experimental.sparse.bcsr_extract)

By The JAX authors

© Copyright 2024, The JAX Authors.\
