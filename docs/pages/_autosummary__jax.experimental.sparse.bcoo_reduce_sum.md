- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.sparse` module](../jax.experimental.sparse.html)
- jax.experimental.sparse.bcoo_reduce_sum

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.sparse.bcoo_reduce_sum.rst "Download source file")
-  .pdf

# jax.experimental.sparse.bcoo_reduce_sum

## Contents

- [`bcoo_reduce_sum()`](#jax.experimental.sparse.bcoo_reduce_sum)

# jax.experimental.sparse.bcoo_reduce_sum[\#](#jax-experimental-sparse-bcoo-reduce-sum "Link to this heading")

jax.experimental.sparse.bcoo_reduce_sum(*mat*, *\**, *axes*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/experimental/sparse/bcoo.py#L2121-L2136)[\#](#jax.experimental.sparse.bcoo_reduce_sum "Link to this definition")  
Sum array element over given axes.

Parameters:  
- **mat** ([*BCOO*](jax.experimental.sparse.BCOO.html#jax.experimental.sparse.BCOO "jax.experimental.sparse.BCOO")) – A BCOO-format array.

- **shape** – The shape of the target array.

- **axes** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – A tuple or list or ndarray which contains axes of `mat` over which sum is performed.

Returns:  
A BCOO-format array containing the result.

Return type:  
[BCOO](jax.experimental.sparse.BCOO.html#jax.experimental.sparse.BCOO "jax.experimental.sparse.BCOO")

[](jax.experimental.sparse.bcoo_update_layout.html "previous page")

previous

jax.experimental.sparse.bcoo_update_layout

[](jax.experimental.sparse.bcoo_reshape.html "next page")

next

jax.experimental.sparse.bcoo_reshape

Contents

- [`bcoo_reduce_sum()`](#jax.experimental.sparse.bcoo_reduce_sum)

By The JAX authors

© Copyright 2024, The JAX Authors.\
