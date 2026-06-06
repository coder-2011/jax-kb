- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.sparse` module](../jax.experimental.sparse.html)
- jax.experimental.sparse.bcoo_transpose

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.sparse.bcoo_transpose.rst "Download source file")
-  .pdf

# jax.experimental.sparse.bcoo_transpose

## Contents

- [`bcoo_transpose()`](#jax.experimental.sparse.bcoo_transpose)

# jax.experimental.sparse.bcoo_transpose[\#](#jax-experimental-sparse-bcoo-transpose "Link to this heading")

jax.experimental.sparse.bcoo_transpose(*mat*, *\**, *permutation*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/experimental/sparse/bcoo.py#L507-L525)[\#](#jax.experimental.sparse.bcoo_transpose "Link to this definition")  
Transpose a BCOO-format array.

Parameters:  
- **mat** ([*BCOO*](jax.experimental.sparse.BCOO.html#jax.experimental.sparse.BCOO "jax.experimental.sparse.BCOO")) – A BCOO-format array.

- **permutation** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – A tuple or list or ndarray which contains a permutation of \[0,1,..,N-1\] where N is the number of axes of `mat` in the order of batch, sparse, and dense dimensions. The i’th axis of the returned array corresponds to the axis numbered permutation\[i\] of `mat`. Transpose permutation currently does not support permuting batch axes with non-batch axes nor permuting dense axes with non-dense axes.

Returns:  
A BCOO-format array.

Return type:  
[BCOO](jax.experimental.sparse.BCOO.html#jax.experimental.sparse.BCOO "jax.experimental.sparse.BCOO")

[](jax.experimental.sparse.bcoo_todense.html "previous page")

previous

jax.experimental.sparse.bcoo_todense

[](jax.experimental.sparse.BCSR.html "next page")

next

jax.experimental.sparse.BCSR

Contents

- [`bcoo_transpose()`](#jax.experimental.sparse.bcoo_transpose)

By The JAX authors

© Copyright 2024, The JAX Authors.\
