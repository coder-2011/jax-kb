- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.sparse` module](../jax.experimental.sparse.html)
- jax.experimental.sparse.CSC

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.sparse.CSC.rst "Download source file")
-  .pdf

# jax.experimental.sparse.CSC

## Contents

- [`CSC`](#jax.experimental.sparse.CSC)
  - [`CSC.__init__()`](#jax.experimental.sparse.CSC.__init__)

# jax.experimental.sparse.CSC[\#](#jax-experimental-sparse-csc "Link to this heading")

*class* jax.experimental.sparse.CSC(*args*, *\**, *shape*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/experimental/sparse/csr.py#L149-L222)[\#](#jax.experimental.sparse.CSC "Link to this definition")  
Experimental CSC matrix implemented in JAX; API subject to change.

Parameters:  
**shape** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*)

\_\_init\_\_(*args*, *\**, *shape*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/experimental/sparse/csr.py#L165-L168)[\#](#jax.experimental.sparse.CSC.__init__ "Link to this definition")  

Methods

|  |  |
|----|----|
| [`__init__`](#jax.experimental.sparse.CSC.__init__ "jax.experimental.sparse.CSC.__init__")(args, \*, shape) |  |
| `block_until_ready`() |  |
| `fromdense`(mat, \*\[, nse, index_dtype\]) |  |
| `sum`(\*args, \*\*kwargs) |  |
| `todense`() |  |
| `transpose`(\[axes\]) |  |
| `tree_flatten`() |  |
| `tree_unflatten`(aux_data, children) |  |

Attributes

|           |     |
|-----------|-----|
| `T`       |     |
| `dtype`   |     |
| `ndim`    |     |
| `nse`     |     |
| `size`    |     |
| `data`    |     |
| `indices` |     |
| `indptr`  |     |
| `shape`   |     |

[](jax.experimental.sparse.COO.html "previous page")

previous

jax.experimental.sparse.COO

[](jax.experimental.sparse.CSR.html "next page")

next

jax.experimental.sparse.CSR

Contents

- [`CSC`](#jax.experimental.sparse.CSC)
  - [`CSC.__init__()`](#jax.experimental.sparse.CSC.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
