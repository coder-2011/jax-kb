- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.sparse` module](../jax.experimental.sparse.html)
- jax.experimental.sparse.empty

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.sparse.empty.rst "Download source file")
-  .pdf

# jax.experimental.sparse.empty

## Contents

- [`empty()`](#jax.experimental.sparse.empty)

# jax.experimental.sparse.empty[\#](#jax-experimental-sparse-empty "Link to this heading")

jax.experimental.sparse.empty(*shape*, *dtype=None*, *index_dtype='int32'*, *sparse_format='bcoo'*, *\*\*kwds*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/experimental/sparse/api.py#L119-L138)[\#](#jax.experimental.sparse.empty "Link to this definition")  
Create an empty sparse array.

Parameters:  
- **shape** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – sequence of integers giving the array shape.

- **dtype** (*DTypeLike* *\|* *None*) – (optional) dtype of the array.

- **index_dtype** (*DTypeLike*) – (optional) dtype of the index arrays.

- **format** – string specifying the matrix format (e.g. \[‘bcoo’\]).

- **\*\*kwds** – additional keywords passed to the format-specific \_empty constructor.

- **sparse_format** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)"))

Returns:  
empty sparse matrix.

Return type:  
mat

[](jax.experimental.sparse.value_and_grad.html "previous page")

previous

jax.experimental.sparse.value_and_grad

[](jax.experimental.sparse.eye.html "next page")

next

jax.experimental.sparse.eye

Contents

- [`empty()`](#jax.experimental.sparse.empty)

By The JAX authors

© Copyright 2024, The JAX Authors.\
