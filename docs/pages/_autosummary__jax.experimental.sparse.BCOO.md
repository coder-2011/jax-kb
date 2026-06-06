- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.sparse` module](../jax.experimental.sparse.html)
- jax.experimental.sparse.BCOO

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.sparse.BCOO.rst "Download source file")
-  .pdf

# jax.experimental.sparse.BCOO

## Contents

- [`BCOO`](#jax.experimental.sparse.BCOO)
  - [`BCOO.data`](#jax.experimental.sparse.BCOO.data)
  - [`BCOO.indices`](#jax.experimental.sparse.BCOO.indices)
  - [`BCOO.__init__()`](#jax.experimental.sparse.BCOO.__init__)

# jax.experimental.sparse.BCOO[\#](#jax-experimental-sparse-bcoo "Link to this heading")

*class* jax.experimental.sparse.BCOO(*args*, *\**, *shape*, *indices_sorted=False*, *unique_indices=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/experimental/sparse/bcoo.py#L2435-L2729)[\#](#jax.experimental.sparse.BCOO "Link to this definition")  
Experimental batched COO matrix implemented in JAX

Parameters:  
- **(data** – data and indices in batched COO format.

- **indices)** – data and indices in batched COO format.

- **shape** (*Shape*) – shape of sparse array.

- **args** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*Array*](jax.Array.html#jax.Array "jax.Array")*,* [*Array*](jax.Array.html#jax.Array "jax.Array")*\]*)

- **indices_sorted** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

- **unique_indices** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

data[\#](#jax.experimental.sparse.BCOO.data "Link to this definition")  
ndarray of shape `[*batch_dims,`` ``nse,`` ``*dense_dims]` containing the explicitly stored data within the sparse matrix.

Type:  
[Array](jax.Array.html#jax.Array "jax.Array")

indices[\#](#jax.experimental.sparse.BCOO.indices "Link to this definition")  
ndarray of shape `[*batch_dims,`` ``nse,`` ``n_sparse]` containing the indices of the explicitly stored data. Duplicate entries will be summed.

Type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Examples

Create a sparse array from a dense array:

    >>> M = jnp.array([[0., 2., 0.], [1., 0., 4.]])
    >>> M_sp = BCOO.fromdense(M)
    >>> M_sp
    BCOO(float32[2, 3], nse=3)

Examine the internal representation:

    >>> M_sp.data
    Array([2., 1., 4.], dtype=float32)
    >>> M_sp.indices
    Array([[0, 1],
           [1, 0],
           [1, 2]], dtype=int32)

Create a dense array from a sparse array:

    >>> M_sp.todense()
    Array([[0., 2., 0.],
           [1., 0., 4.]], dtype=float32)

Create a sparse array from COO data & indices:

    >>> data = jnp.array([1., 3., 5.])
    >>> indices = jnp.array([[0, 0],
    ...                      [1, 1],
    ...                      [2, 2]])
    >>> mat = BCOO((data, indices), shape=(3, 3))
    >>> mat
    BCOO(float32[3, 3], nse=3)
    >>> mat.todense()
    Array([[1., 0., 0.],
           [0., 3., 0.],
           [0., 0., 5.]], dtype=float32)

\_\_init\_\_(*args*, *\**, *shape*, *indices_sorted=False*, *unique_indices=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/experimental/sparse/bcoo.py#L2494-L2501)[\#](#jax.experimental.sparse.BCOO.__init__ "Link to this definition")  
Parameters:  
- **args** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*Array*](jax.Array.html#jax.Array "jax.Array")*,* [*Array*](jax.Array.html#jax.Array "jax.Array")*\]*)

- **shape** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*)

- **indices_sorted** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

- **unique_indices** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

Methods

|  |  |
|----|----|
| [`__init__`](#jax.experimental.sparse.BCOO.__init__ "jax.experimental.sparse.BCOO.__init__")(args, \*, shape\[, indices_sorted, ...\]) |  |
| `astype`(\*args, \*\*kwargs) | Copy the array and cast to a specified dtype. |
| `block_until_ready`() |  |
| `from_scipy_sparse`(mat, \*\[, index_dtype, ...\]) | Create a BCOO array from a [`scipy.sparse`](https://scipy.github.io/devdocs/reference/sparse.html#module-scipy.sparse "(in SciPy v1.19.0.dev)") array. |
| `fromdense`(mat, \*\[, nse, index_dtype, ...\]) | Create a BCOO array from a (dense) [`Array`](jax.Array.html#jax.Array "jax.Array"). |
| `reshape`(\*args, \*\*kwargs) | Returns an array containing the same data with a new shape. |
| `sort_indices`() | Return a copy of the matrix with indices sorted. |
| `sum`(\*args, \*\*kwargs) | Sum array along axis. |
| `sum_duplicates`(\[nse, remove_zeros\]) | Return a copy of the array with duplicate indices summed. |
| `todense`() | Create a dense version of the array. |
| `transpose`(\[axes\]) | Create a new array containing the transpose. |
| `tree_flatten`() |  |
| `tree_unflatten`(aux_data, children) |  |
| `update_layout`(\*\[, n_batch, n_dense, ...\]) | Update the storage layout (i.e. n_batch & n_dense) of a BCOO matrix. |

Attributes

|  |  |
|----|----|
| `T` |  |
| `dtype` |  |
| `n_batch` |  |
| `n_dense` |  |
| `n_sparse` |  |
| `ndim` |  |
| `nse` |  |
| `size` |  |
| [`data`](#jax.experimental.sparse.BCOO.data "jax.experimental.sparse.BCOO.data") |  |
| [`indices`](#jax.experimental.sparse.BCOO.indices "jax.experimental.sparse.BCOO.indices") |  |
| `shape` |  |
| `indices_sorted` |  |
| `unique_indices` |  |

[](jax.experimental.sparse.JAXSparse.html "previous page")

previous

jax.experimental.sparse.JAXSparse

[](jax.experimental.sparse.bcoo_broadcast_in_dim.html "next page")

next

jax.experimental.sparse.bcoo_broadcast_in_dim

Contents

- [`BCOO`](#jax.experimental.sparse.BCOO)
  - [`BCOO.data`](#jax.experimental.sparse.BCOO.data)
  - [`BCOO.indices`](#jax.experimental.sparse.BCOO.indices)
  - [`BCOO.__init__()`](#jax.experimental.sparse.BCOO.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
