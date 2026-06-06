- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.sparse` module](../jax.experimental.sparse.html)
- jax.experimental.sparse.COO

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.sparse.COO.rst "Download source file")
-  .pdf

# jax.experimental.sparse.COO

## Contents

- [`COO`](#jax.experimental.sparse.COO)
  - [`COO.__init__()`](#jax.experimental.sparse.COO.__init__)

# jax.experimental.sparse.COO[\#](#jax-experimental-sparse-coo "Link to this heading")

*class* jax.experimental.sparse.COO(*args*, *\**, *shape*, *rows_sorted=False*, *cols_sorted=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/experimental/sparse/coo.py#L52-L180)[\#](#jax.experimental.sparse.COO "Link to this definition")  
Experimental COO matrix implemented in JAX.

Note: this class has minimal compatibility with JAX transforms such as grad and autodiff, and offers very little functionality. In general you should prefer [`jax.experimental.sparse.BCOO`](jax.experimental.sparse.BCOO.html#jax.experimental.sparse.BCOO "jax.experimental.sparse.BCOO").

Additionally, there are known failures in the case that nse is larger than the true number of nonzeros in the represented matrix. This situation is better handled in BCOO.

Parameters:  
- **args** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*Array*](jax.Array.html#jax.Array "jax.Array")*,* [*Array*](jax.Array.html#jax.Array "jax.Array")*,* [*Array*](jax.Array.html#jax.Array "jax.Array")*\]*)

- **shape** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*)

- **rows_sorted** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

- **cols_sorted** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

\_\_init\_\_(*args*, *\**, *shape*, *rows_sorted=False*, *cols_sorted=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/experimental/sparse/coo.py#L89-L95)[\#](#jax.experimental.sparse.COO.__init__ "Link to this definition")  
Parameters:  
- **args** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*Array*](jax.Array.html#jax.Array "jax.Array")*,* [*Array*](jax.Array.html#jax.Array "jax.Array")*,* [*Array*](jax.Array.html#jax.Array "jax.Array")*\]*)

- **shape** (*Shape*)

- **rows_sorted** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

- **cols_sorted** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

Methods

|  |  |
|----|----|
| [`__init__`](#jax.experimental.sparse.COO.__init__ "jax.experimental.sparse.COO.__init__")(args, \*, shape\[, rows_sorted, ...\]) |  |
| `block_until_ready`() |  |
| `fromdense`(mat, \*\[, nse, index_dtype\]) |  |
| `sum`(\*args, \*\*kwargs) |  |
| `todense`() |  |
| `transpose`(\[axes\]) |  |
| `tree_flatten`() |  |
| `tree_unflatten`(aux_data, children) |  |

Attributes

|         |     |
|---------|-----|
| `T`     |     |
| `dtype` |     |
| `ndim`  |     |
| `nse`   |     |
| `size`  |     |
| `data`  |     |
| `row`   |     |
| `col`   |     |
| `shape` |     |

[](jax.experimental.sparse.bcsr_todense.html "previous page")

previous

jax.experimental.sparse.bcsr_todense

[](jax.experimental.sparse.CSC.html "next page")

next

jax.experimental.sparse.CSC

Contents

- [`COO`](#jax.experimental.sparse.COO)
  - [`COO.__init__()`](#jax.experimental.sparse.COO.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
