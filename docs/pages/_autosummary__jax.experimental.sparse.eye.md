- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.sparse` module](../jax.experimental.sparse.html)
- jax.experimental.sparse.eye

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.sparse.eye.rst "Download source file")
-  .pdf

# jax.experimental.sparse.eye

## Contents

- [`eye()`](#jax.experimental.sparse.eye)

# jax.experimental.sparse.eye[\#](#jax-experimental-sparse-eye "Link to this heading")

jax.experimental.sparse.eye(*N*, *M=None*, *k=0*, *dtype=None*, *index_dtype='int32'*, *sparse_format='bcoo'*, *\*\*kwds*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/experimental/sparse/api.py#L140-L167)[\#](#jax.experimental.sparse.eye "Link to this definition")  
Create 2D sparse identity matrix.

Parameters:  
- **N** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – int. Number of rows in the output.

- **M** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – int, optional. Number of columns in the output. If None, defaults to N.

- **k** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – int, optional. Index of the diagonal: 0 (the default) refers to the main diagonal, a positive value refers to an upper diagonal, and a negative value to a lower diagonal.

- **dtype** (*DTypeLike* *\|* *None*) – data-type, optional. Data-type of the returned array.

- **index_dtype** (*DTypeLike*) – (optional) dtype of the index arrays.

- **format** – string specifying the matrix format (e.g. \[‘bcoo’\]).

- **\*\*kwds** – additional keywords passed to the format-specific \_empty constructor.

- **sparse_format** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)"))

Returns:  
two-dimensional sparse matrix with ones along the k-th diagonal.

Return type:  
I

[](jax.experimental.sparse.empty.html "previous page")

previous

jax.experimental.sparse.empty

[](jax.experimental.sparse.todense.html "next page")

next

jax.experimental.sparse.todense

Contents

- [`eye()`](#jax.experimental.sparse.eye)

By The JAX authors

© Copyright 2024, The JAX Authors.\
