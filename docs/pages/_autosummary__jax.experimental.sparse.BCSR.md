- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.sparse` module](../jax.experimental.sparse.html)
- jax.experimental.sparse.BCSR

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.sparse.BCSR.rst "Download source file")
-  .pdf

# jax.experimental.sparse.BCSR

## Contents

- [`BCSR`](#jax.experimental.sparse.BCSR)
  - [`BCSR.__init__()`](#jax.experimental.sparse.BCSR.__init__)

# jax.experimental.sparse.BCSR[\#](#jax-experimental-sparse-bcsr "Link to this heading")

*class* jax.experimental.sparse.BCSR(*args*, *\**, *shape*, *indices_sorted=False*, *unique_indices=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/experimental/sparse/bcsr.py#L832-L1002)[\#](#jax.experimental.sparse.BCSR "Link to this definition")  
Experimental batched CSR matrix implemented in JAX.

Parameters:  
- **args** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*Array*](jax.Array.html#jax.Array "jax.Array")*,* [*Array*](jax.Array.html#jax.Array "jax.Array")*,* [*Array*](jax.Array.html#jax.Array "jax.Array")*\]*)

- **shape** (*Shape*)

- **indices_sorted** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

- **unique_indices** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

\_\_init\_\_(*args*, *\**, *shape*, *indices_sorted=False*, *unique_indices=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/experimental/sparse/bcsr.py#L876-L883)[\#](#jax.experimental.sparse.BCSR.__init__ "Link to this definition")  
Parameters:  
- **args** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*Array*](jax.Array.html#jax.Array "jax.Array")*,* [*Array*](jax.Array.html#jax.Array "jax.Array")*,* [*Array*](jax.Array.html#jax.Array "jax.Array")*\]*)

- **shape** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*)

- **indices_sorted** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

- **unique_indices** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

Methods

|  |  |
|----|----|
| [`__init__`](#jax.experimental.sparse.BCSR.__init__ "jax.experimental.sparse.BCSR.__init__")(args, \*, shape\[, indices_sorted, ...\]) |  |
| `block_until_ready`() |  |
| `from_bcoo`(arr) |  |
| `from_scipy_sparse`(mat, \*\[, index_dtype, ...\]) | Create a BCSR array from a [`scipy.sparse`](https://scipy.github.io/devdocs/reference/sparse.html#module-scipy.sparse "(in SciPy v1.19.0.dev)") array. |
| `fromdense`(mat, \*\[, nse, index_dtype, ...\]) | Create a BCSR array from a (dense) `Array`. |
| `sum`(\*args, \*\*kwargs) |  |
| `sum_duplicates`(\[nse, remove_zeros\]) | Return a copy of the array with duplicate indices summed. |
| `to_bcoo`() |  |
| `todense`() | Create a dense version of the array. |
| `transpose`(\*args, \*\*kwargs) |  |
| `tree_flatten`() |  |
| `tree_unflatten`(aux_data, children) |  |

Attributes

|                  |     |
|------------------|-----|
| `T`              |     |
| `dtype`          |     |
| `n_batch`        |     |
| `n_dense`        |     |
| `n_sparse`       |     |
| `ndim`           |     |
| `nse`            |     |
| `size`           |     |
| `data`           |     |
| `indices`        |     |
| `indptr`         |     |
| `shape`          |     |
| `indices_sorted` |     |
| `unique_indices` |     |

[](jax.experimental.sparse.bcoo_transpose.html "previous page")

previous

jax.experimental.sparse.bcoo_transpose

[](jax.experimental.sparse.bcsr_dot_general.html "next page")

next

jax.experimental.sparse.bcsr_dot_general

Contents

- [`BCSR`](#jax.experimental.sparse.BCSR)
  - [`BCSR.__init__()`](#jax.experimental.sparse.BCSR.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
