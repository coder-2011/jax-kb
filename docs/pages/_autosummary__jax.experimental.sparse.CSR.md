- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.sparse` module](../jax.experimental.sparse.html)
- jax.experimental.sparse.CSR

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.sparse.CSR.rst "Download source file")
-  .pdf

# jax.experimental.sparse.CSR

## Contents

- [`CSR`](#jax.experimental.sparse.CSR)
  - [`CSR.__init__()`](#jax.experimental.sparse.CSR.__init__)

# jax.experimental.sparse.CSR[\#](#jax-experimental-sparse-csr "Link to this heading")

*class* jax.experimental.sparse.CSR(*args*, *\**, *shape*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/experimental/sparse/csr.py#L44-L147)[\#](#jax.experimental.sparse.CSR "Link to this definition")  
Experimental CSR matrix implemented in JAX.

Note: this class has minimal compatibility with JAX transforms such as grad and autodiff, and offers very little functionality. In general you should prefer [`jax.experimental.sparse.BCOO`](jax.experimental.sparse.BCOO.html#jax.experimental.sparse.BCOO "jax.experimental.sparse.BCOO").

Additionally, there are known failures in the case that nse is larger than the true number of nonzeros in the represented matrix. This situation is better handled in BCOO.

Parameters:  
**shape** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*)

\_\_init\_\_(*args*, *\**, *shape*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/experimental/sparse/csr.py#L73-L76)[\#](#jax.experimental.sparse.CSR.__init__ "Link to this definition")  

Methods

|  |  |
|----|----|
| [`__init__`](#jax.experimental.sparse.CSR.__init__ "jax.experimental.sparse.CSR.__init__")(args, \*, shape) |  |
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

[](jax.experimental.sparse.CSC.html "previous page")

previous

jax.experimental.sparse.CSC

[](jax.experimental.sparse.coo_fromdense.html "next page")

next

jax.experimental.sparse.coo_fromdense

Contents

- [`CSR`](#jax.experimental.sparse.CSR)
  - [`CSR.__init__()`](#jax.experimental.sparse.CSR.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
