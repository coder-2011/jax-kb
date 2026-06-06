- [](index.html)
- [API Reference](jax.html)
- [`jax.experimental` module](jax.experimental.html)
- `jax.experimental.sparse` module

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](_sources/jax.experimental.sparse.rst "Download source file")
-  .pdf

# jax.experimental.sparse module

## Contents

- [Batched-coordinate (BCOO) sparse matrices](#batched-coordinate-bcoo-sparse-matrices)
- [Sparsify transform](#sparsify-transform)
- [Example: sparse logistic regression](#example-sparse-logistic-regression)
- [Sparse API Reference](#sparse-api-reference)
  - [BCOO Data Structure](#bcoo-data-structure)
  - [BCSR Data Structure](#bcsr-data-structure)
  - [Other Sparse Data Structures](#other-sparse-data-structures)
  - [`jax.experimental.sparse.linalg`](#module-jax.experimental.sparse.linalg)

# `jax.experimental.sparse` module[\#](#module-jax.experimental.sparse "Link to this heading")

Note

The methods in `jax.experimental.sparse` are experimental reference implementations, and not recommended for use in performance-critical applications. The submodule is no longer being actively developed, but the team will continue supporting existing features as best we can.

The [`jax.experimental.sparse`](#module-jax.experimental.sparse "jax.experimental.sparse") module includes experimental support for sparse matrix operations in JAX. The primary interfaces made available are the [`BCOO`](_autosummary/jax.experimental.sparse.BCOO.html#jax.experimental.sparse.BCOO "jax.experimental.sparse.BCOO") sparse array type, and the [`sparsify()`](_autosummary/jax.experimental.sparse.sparsify.html#jax.experimental.sparse.sparsify "jax.experimental.sparse.sparsify") transform.

## Batched-coordinate (BCOO) sparse matrices[\#](#batched-coordinate-bcoo-sparse-matrices "Link to this heading")

The main high-level sparse object currently available in JAX is the [`BCOO`](_autosummary/jax.experimental.sparse.BCOO.html#jax.experimental.sparse.BCOO "jax.experimental.sparse.BCOO"), or *batched coordinate* sparse array, which offers a compressed storage format compatible with JAX transformations, in particular JIT (e.g. [`jax.jit()`](_autosummary/jax.jit.html#jax.jit "jax.jit")), batching (e.g. [`jax.vmap()`](_autosummary/jax.vmap.html#jax.vmap "jax.vmap")) and autodiff (e.g. [`jax.grad()`](_autosummary/jax.grad.html#jax.grad "jax.grad")).

Here is an example of creating a sparse array from a dense array:

    >>> from jax.experimental import sparse
    >>> import jax.numpy as jnp
    >>> import numpy as np

    >>> M = jnp.array([[0., 1., 0., 2.],
    ...                [3., 0., 0., 0.],
    ...                [0., 0., 4., 0.]])

    >>> M_sp = sparse.BCOO.fromdense(M)

    >>> M_sp
    BCOO(float32[3, 4], nse=4)

Convert back to a dense array with the `todense()` method:

    >>> M_sp.todense()
    Array([[0., 1., 0., 2.],
           [3., 0., 0., 0.],
           [0., 0., 4., 0.]], dtype=float32)

The BCOO format is a somewhat modified version of the standard COO format, and the dense representation can be seen in the `data` and `indices` attributes:

    >>> M_sp.data  # Explicitly stored data
    Array([1., 2., 3., 4.], dtype=float32)

    >>> M_sp.indices # Indices of the stored data
    Array([[0, 1],
           [0, 3],
           [1, 0],
           [2, 2]], dtype=int32)

BCOO objects have familiar array-like attributes, as well as sparse-specific attributes:

    >>> M_sp.ndim
    2

    >>> M_sp.shape
    (3, 4)

    >>> M_sp.dtype
    dtype('float32')

    >>> M_sp.nse  # "number of specified elements"
    4

BCOO objects also implement a number of array-like methods, to allow you to use them directly within jax programs. For example, here we compute the transposed matrix-vector product:

    >>> y = jnp.array([3., 6., 5.])

    >>> M_sp.T @ y
    Array([18.,  3., 20.,  6.], dtype=float32)

    >>> M.T @ y  # Compare to dense version
    Array([18.,  3., 20.,  6.], dtype=float32)

BCOO objects are designed to be compatible with JAX transforms, including [`jax.jit()`](_autosummary/jax.jit.html#jax.jit "jax.jit"), [`jax.vmap()`](_autosummary/jax.vmap.html#jax.vmap "jax.vmap"), [`jax.grad()`](_autosummary/jax.grad.html#jax.grad "jax.grad"), and others. For example:

    >>> from jax import grad, jit

    >>> def f(y):
    ...   return (M_sp.T @ y).sum()
    ...
    >>> jit(grad(f))(y)
    Array([3., 3., 4.], dtype=float32)

Note, however, that under normal circumstances [`jax.numpy`](jax.numpy.html#module-jax.numpy "jax.numpy") and [`jax.lax`](jax.lax.html#module-jax.lax "jax.lax") functions do not know how to handle sparse matrices, so attempting to compute things like `jnp.dot(M_sp.T,`` ``y)` will result in an error (however, see the next section).

## Sparsify transform[\#](#sparsify-transform "Link to this heading")

An overarching goal of the JAX sparse implementation is to provide a means to switch from dense to sparse computation seamlessly, without having to modify the dense implementation. This sparse experiment accomplishes this through the [`sparsify()`](_autosummary/jax.experimental.sparse.sparsify.html#jax.experimental.sparse.sparsify "jax.experimental.sparse.sparsify") transform.

Consider this function, which computes a more complicated result from a matrix and a vector input:

    >>> def f(M, v):
    ...   return 2 * jnp.dot(jnp.log1p(M.T), v) + 1
    ...
    >>> f(M, y)
    Array([17.635532,  5.158883, 17.09438 ,  7.591674], dtype=float32)

Were we to pass a sparse matrix to this directly, it would result in an error, because `jnp` functions do not recognize sparse inputs. However, with [`sparsify()`](_autosummary/jax.experimental.sparse.sparsify.html#jax.experimental.sparse.sparsify "jax.experimental.sparse.sparsify"), we get a version of this function that does accept sparse matrices:

    >>> f_sp = sparse.sparsify(f)

    >>> f_sp(M_sp, y)
    Array([17.635532,  5.158883, 17.09438 ,  7.591674], dtype=float32)

Support for [`sparsify()`](_autosummary/jax.experimental.sparse.sparsify.html#jax.experimental.sparse.sparsify "jax.experimental.sparse.sparsify") includes a large number of the most common primitives, including:

- generalized (batched) matrix products & einstein summations (`dot_general_p`)

- zero-preserving elementwise binary operations (e.g. `add_p`, `mul_p`, etc.)

- zero-preserving elementwise unary operations (e.g. `abs_p`, `jax.lax.neg_p`, etc.)

- summation reductions (`reduce_sum_p`)

- general indexing operations (`slice_p`, lax.dynamic_slice_p, lax.gather_p)

- concatenation and stacking (`concatenate_p`)

- transposition & reshaping ((`transpose_p`, `reshape_p`, `squeeze_p`, `broadcast_in_dim_p`)

- some higher-order functions (`cond_p`, `while_p`, `scan_p`)

- some simple 1D convolutions (`conv_general_dilated_p`)

Nearly any [`jax.numpy`](jax.numpy.html#module-jax.numpy "jax.numpy") function that lowers to these supported primitives can be used within a sparsify transform to operate on sparse arrays. This set of primitives is enough to enable relatively sophisticated sparse workflows, as the next section will show.

## Example: sparse logistic regression[\#](#example-sparse-logistic-regression "Link to this heading")

As an example of a more complicated sparse workflow, let’s consider a simple logistic regression implemented in JAX. Notice that the following implementation has no reference to sparsity:

    >>> import functools
    >>> from sklearn.datasets import make_classification
    >>> from jax.scipy import optimize

    >>> def sigmoid(x):
    ...   return 0.5 * (jnp.tanh(x / 2) + 1)
    ...
    >>> def y_model(params, X):
    ...   return sigmoid(jnp.dot(X, params[1:]) + params[0])
    ...
    >>> def loss(params, X, y):
    ...   y_hat = y_model(params, X)
    ...   return -jnp.mean(y * jnp.log(y_hat) + (1 - y) * jnp.log(1 - y_hat))
    ...
    >>> def fit_logreg(X, y):
    ...   params = jnp.zeros(X.shape[1] + 1)
    ...   result = optimize.minimize(functools.partial(loss, X=X, y=y),
    ...                              x0=params, method='BFGS')
    ...   return result.x

    >>> X, y = make_classification(n_classes=2, random_state=1701)
    >>> params_dense = fit_logreg(X, y)
    >>> print(params_dense)  
    [-0.7298445   0.29893667  1.0248291  -0.44436368  0.8785025  -0.7724008
     -0.62893456  0.2934014   0.82974285  0.16838408 -0.39774987 -0.5071844
      0.2028872   0.5227761  -0.3739224  -0.7104083   2.4212713   0.6310087
     -0.67060554  0.03139788 -0.05359547]

This returns the best-fit parameters of a dense logistic regression problem. To fit the same model on sparse data, we can apply the [`sparsify()`](_autosummary/jax.experimental.sparse.sparsify.html#jax.experimental.sparse.sparsify "jax.experimental.sparse.sparsify") transform:

    >>> Xsp = sparse.BCOO.fromdense(X)  # Sparse version of the input
    >>> fit_logreg_sp = sparse.sparsify(fit_logreg)  # Sparse-transformed fit function
    >>> params_sparse = fit_logreg_sp(Xsp, y)
    >>> print(params_sparse)  
    [-0.72971725  0.29878938  1.0246326  -0.44430563  0.8784217  -0.77225566
     -0.6288222   0.29335397  0.8293481   0.16820715 -0.39764675 -0.5069753
      0.202579    0.522672   -0.3740134  -0.7102678   2.4209507   0.6310593
     -0.670236    0.03132951 -0.05356663]

## Sparse API Reference[\#](#sparse-api-reference "Link to this heading")

|  |  |
|----|----|
| [`sparsify`](_autosummary/jax.experimental.sparse.sparsify.html#jax.experimental.sparse.sparsify "jax.experimental.sparse.sparsify")(f\[, use_tracer\]) | Experimental sparsification transform. |
| [`grad`](_autosummary/jax.experimental.sparse.grad.html#jax.experimental.sparse.grad "jax.experimental.sparse.grad")(fun\[, argnums, has_aux\]) | Sparse-aware version of [`jax.grad()`](_autosummary/jax.grad.html#jax.grad "jax.grad") |
| [`value_and_grad`](_autosummary/jax.experimental.sparse.value_and_grad.html#jax.experimental.sparse.value_and_grad "jax.experimental.sparse.value_and_grad")(fun\[, argnums, has_aux\]) | Sparse-aware version of [`jax.value_and_grad()`](_autosummary/jax.value_and_grad.html#jax.value_and_grad "jax.value_and_grad") |
| [`empty`](_autosummary/jax.experimental.sparse.empty.html#jax.experimental.sparse.empty "jax.experimental.sparse.empty")(shape\[, dtype, index_dtype, sparse_format\]) | Create an empty sparse array. |
| [`eye`](_autosummary/jax.experimental.sparse.eye.html#jax.experimental.sparse.eye "jax.experimental.sparse.eye")(N\[, M, k, dtype, index_dtype, sparse_format\]) | Create 2D sparse identity matrix. |
| [`todense`](_autosummary/jax.experimental.sparse.todense.html#jax.experimental.sparse.todense "jax.experimental.sparse.todense")(arr) | Convert input to a dense matrix. |
| [`random_bcoo`](_autosummary/jax.experimental.sparse.random_bcoo.html#jax.experimental.sparse.random_bcoo "jax.experimental.sparse.random_bcoo")(key, shape, \*\[, dtype, ...\]) | Generate a random BCOO matrix. |
| [`JAXSparse`](_autosummary/jax.experimental.sparse.JAXSparse.html#jax.experimental.sparse.JAXSparse "jax.experimental.sparse.JAXSparse")(args, \*, shape) | Base class for high-level JAX sparse objects. |

### BCOO Data Structure[\#](#bcoo-data-structure "Link to this heading")

[`BCOO`](_autosummary/jax.experimental.sparse.BCOO.html#jax.experimental.sparse.BCOO "jax.experimental.sparse.BCOO") is the *Batched COO format*, and is the main sparse data structure implemented in [`jax.experimental.sparse`](#module-jax.experimental.sparse "jax.experimental.sparse"). Its operations are compatible with JAX’s core transformations, including batching (e.g. [`jax.vmap()`](_autosummary/jax.vmap.html#jax.vmap "jax.vmap")) and autodiff (e.g. [`jax.grad()`](_autosummary/jax.grad.html#jax.grad "jax.grad")).

|  |  |
|----|----|
| [`BCOO`](_autosummary/jax.experimental.sparse.BCOO.html#jax.experimental.sparse.BCOO "jax.experimental.sparse.BCOO")(args, \*, shape\[, indices_sorted, ...\]) | Experimental batched COO matrix implemented in JAX |
| [`bcoo_broadcast_in_dim`](_autosummary/jax.experimental.sparse.bcoo_broadcast_in_dim.html#jax.experimental.sparse.bcoo_broadcast_in_dim "jax.experimental.sparse.bcoo_broadcast_in_dim")(mat, \*, shape, ...\[, ...\]) | Expand the size and rank of a BCOO array by duplicating the data. |
| [`bcoo_concatenate`](_autosummary/jax.experimental.sparse.bcoo_concatenate.html#jax.experimental.sparse.bcoo_concatenate "jax.experimental.sparse.bcoo_concatenate")(operands, \*, dimension) | Sparse implementation of [`jax.lax.concatenate()`](_autosummary/jax.lax.concatenate.html#jax.lax.concatenate "jax.lax.concatenate") |
| [`bcoo_dot_general`](_autosummary/jax.experimental.sparse.bcoo_dot_general.html#jax.experimental.sparse.bcoo_dot_general "jax.experimental.sparse.bcoo_dot_general")(lhs, rhs, \*, dimension_numbers) | A general contraction operation. |
| [`bcoo_dot_general_sampled`](_autosummary/jax.experimental.sparse.bcoo_dot_general_sampled.html#jax.experimental.sparse.bcoo_dot_general_sampled "jax.experimental.sparse.bcoo_dot_general_sampled")(A, B, indices, \*, ...) | A contraction operation with output computed at given sparse indices. |
| [`bcoo_dynamic_slice`](_autosummary/jax.experimental.sparse.bcoo_dynamic_slice.html#jax.experimental.sparse.bcoo_dynamic_slice "jax.experimental.sparse.bcoo_dynamic_slice")(mat, start_indices, ...) | Sparse implementation of [`jax.lax.dynamic_slice()`](_autosummary/jax.lax.dynamic_slice.html#jax.lax.dynamic_slice "jax.lax.dynamic_slice"). |
| [`bcoo_extract`](_autosummary/jax.experimental.sparse.bcoo_extract.html#jax.experimental.sparse.bcoo_extract "jax.experimental.sparse.bcoo_extract")(sparr, arr, \*\[, assume_unique\]) | Extract values from a dense array according to the sparse array's indices. |
| [`bcoo_fromdense`](_autosummary/jax.experimental.sparse.bcoo_fromdense.html#jax.experimental.sparse.bcoo_fromdense "jax.experimental.sparse.bcoo_fromdense")(mat, \*\[, nse, n_batch, ...\]) | Create BCOO-format sparse matrix from a dense matrix. |
| [`bcoo_gather`](_autosummary/jax.experimental.sparse.bcoo_gather.html#jax.experimental.sparse.bcoo_gather "jax.experimental.sparse.bcoo_gather")(operand, start_indices, ...\[, ...\]) | BCOO version of lax.gather. |
| [`bcoo_multiply_dense`](_autosummary/jax.experimental.sparse.bcoo_multiply_dense.html#jax.experimental.sparse.bcoo_multiply_dense "jax.experimental.sparse.bcoo_multiply_dense")(sp_mat, v) | An element-wise multiplication between a sparse and a dense array. |
| [`bcoo_multiply_sparse`](_autosummary/jax.experimental.sparse.bcoo_multiply_sparse.html#jax.experimental.sparse.bcoo_multiply_sparse "jax.experimental.sparse.bcoo_multiply_sparse")(lhs, rhs) | An element-wise multiplication of two sparse arrays. |
| [`bcoo_update_layout`](_autosummary/jax.experimental.sparse.bcoo_update_layout.html#jax.experimental.sparse.bcoo_update_layout "jax.experimental.sparse.bcoo_update_layout")(mat, \*\[, n_batch, ...\]) | Update the storage layout (i.e. n_batch & n_dense) of a BCOO matrix. |
| [`bcoo_reduce_sum`](_autosummary/jax.experimental.sparse.bcoo_reduce_sum.html#jax.experimental.sparse.bcoo_reduce_sum "jax.experimental.sparse.bcoo_reduce_sum")(mat, \*, axes) | Sum array element over given axes. |
| [`bcoo_reshape`](_autosummary/jax.experimental.sparse.bcoo_reshape.html#jax.experimental.sparse.bcoo_reshape "jax.experimental.sparse.bcoo_reshape")(mat, \*, new_sizes\[, ...\]) | Sparse implementation of [`jax.lax.reshape()`](_autosummary/jax.lax.reshape.html#jax.lax.reshape "jax.lax.reshape"). |
| [`bcoo_slice`](_autosummary/jax.experimental.sparse.bcoo_slice.html#jax.experimental.sparse.bcoo_slice "jax.experimental.sparse.bcoo_slice")(mat, \*, start_indices, limit_indices) | Sparse implementation of [`jax.lax.slice()`](_autosummary/jax.lax.slice.html#jax.lax.slice "jax.lax.slice"). |
| [`bcoo_sort_indices`](_autosummary/jax.experimental.sparse.bcoo_sort_indices.html#jax.experimental.sparse.bcoo_sort_indices "jax.experimental.sparse.bcoo_sort_indices")(mat) | Sort indices of a BCOO array. |
| [`bcoo_squeeze`](_autosummary/jax.experimental.sparse.bcoo_squeeze.html#jax.experimental.sparse.bcoo_squeeze "jax.experimental.sparse.bcoo_squeeze")(arr, \*, dimensions) | Sparse implementation of [`jax.lax.squeeze()`](_autosummary/jax.lax.squeeze.html#jax.lax.squeeze "jax.lax.squeeze"). |
| [`bcoo_sum_duplicates`](_autosummary/jax.experimental.sparse.bcoo_sum_duplicates.html#jax.experimental.sparse.bcoo_sum_duplicates "jax.experimental.sparse.bcoo_sum_duplicates")(mat\[, nse\]) | Sums duplicate indices within a BCOO array, returning an array with sorted indices. |
| [`bcoo_todense`](_autosummary/jax.experimental.sparse.bcoo_todense.html#jax.experimental.sparse.bcoo_todense "jax.experimental.sparse.bcoo_todense")(mat) | Convert batched sparse matrix to a dense matrix. |
| [`bcoo_transpose`](_autosummary/jax.experimental.sparse.bcoo_transpose.html#jax.experimental.sparse.bcoo_transpose "jax.experimental.sparse.bcoo_transpose")(mat, \*, permutation) | Transpose a BCOO-format array. |

### BCSR Data Structure[\#](#bcsr-data-structure "Link to this heading")

[`BCSR`](_autosummary/jax.experimental.sparse.BCSR.html#jax.experimental.sparse.BCSR "jax.experimental.sparse.BCSR") is the *Batched Compressed Sparse Row* format, and is under development. Its operations are compatible with JAX’s core transformations, including batching (e.g. [`jax.vmap()`](_autosummary/jax.vmap.html#jax.vmap "jax.vmap")) and autodiff (e.g. [`jax.grad()`](_autosummary/jax.grad.html#jax.grad "jax.grad")).

|  |  |
|----|----|
| [`BCSR`](_autosummary/jax.experimental.sparse.BCSR.html#jax.experimental.sparse.BCSR "jax.experimental.sparse.BCSR")(args, \*, shape\[, indices_sorted, ...\]) | Experimental batched CSR matrix implemented in JAX. |
| [`bcsr_dot_general`](_autosummary/jax.experimental.sparse.bcsr_dot_general.html#jax.experimental.sparse.bcsr_dot_general "jax.experimental.sparse.bcsr_dot_general")(lhs, rhs, \*, dimension_numbers) | A general contraction operation. |
| [`bcsr_extract`](_autosummary/jax.experimental.sparse.bcsr_extract.html#jax.experimental.sparse.bcsr_extract "jax.experimental.sparse.bcsr_extract")(indices, indptr, mat) | Extract values from a dense matrix at given BCSR (indices, indptr). |
| [`bcsr_fromdense`](_autosummary/jax.experimental.sparse.bcsr_fromdense.html#jax.experimental.sparse.bcsr_fromdense "jax.experimental.sparse.bcsr_fromdense")(mat, \*\[, nse, n_batch, ...\]) | Create BCSR-format sparse matrix from a dense matrix. |
| [`bcsr_todense`](_autosummary/jax.experimental.sparse.bcsr_todense.html#jax.experimental.sparse.bcsr_todense "jax.experimental.sparse.bcsr_todense")(mat) | Convert batched sparse matrix to a dense matrix. |

### Other Sparse Data Structures[\#](#other-sparse-data-structures "Link to this heading")

Other sparse data structures include [`COO`](_autosummary/jax.experimental.sparse.COO.html#jax.experimental.sparse.COO "jax.experimental.sparse.COO"), [`CSR`](_autosummary/jax.experimental.sparse.CSR.html#jax.experimental.sparse.CSR "jax.experimental.sparse.CSR"), and [`CSC`](_autosummary/jax.experimental.sparse.CSC.html#jax.experimental.sparse.CSC "jax.experimental.sparse.CSC"). These are reference implementations of simple sparse structures with a few core operations implemented. Their operations are generally compatible with autodiff transformations such as [`jax.grad()`](_autosummary/jax.grad.html#jax.grad "jax.grad"), but not with batching transforms like [`jax.vmap()`](_autosummary/jax.vmap.html#jax.vmap "jax.vmap").

|  |  |
|----|----|
| [`COO`](_autosummary/jax.experimental.sparse.COO.html#jax.experimental.sparse.COO "jax.experimental.sparse.COO")(args, \*, shape\[, rows_sorted, cols_sorted\]) | Experimental COO matrix implemented in JAX. |
| [`CSC`](_autosummary/jax.experimental.sparse.CSC.html#jax.experimental.sparse.CSC "jax.experimental.sparse.CSC")(args, \*, shape) | Experimental CSC matrix implemented in JAX; API subject to change. |
| [`CSR`](_autosummary/jax.experimental.sparse.CSR.html#jax.experimental.sparse.CSR "jax.experimental.sparse.CSR")(args, \*, shape) | Experimental CSR matrix implemented in JAX. |
| [`coo_fromdense`](_autosummary/jax.experimental.sparse.coo_fromdense.html#jax.experimental.sparse.coo_fromdense "jax.experimental.sparse.coo_fromdense")(mat, \*\[, nse, index_dtype\]) | Create a COO-format sparse matrix from a dense matrix. |
| [`coo_matmat`](_autosummary/jax.experimental.sparse.coo_matmat.html#jax.experimental.sparse.coo_matmat "jax.experimental.sparse.coo_matmat")(mat, B, \*\[, transpose\]) | Product of COO sparse matrix and a dense matrix. |
| [`coo_matvec`](_autosummary/jax.experimental.sparse.coo_matvec.html#jax.experimental.sparse.coo_matvec "jax.experimental.sparse.coo_matvec")(mat, v\[, transpose\]) | Product of COO sparse matrix and a dense vector. |
| [`coo_todense`](_autosummary/jax.experimental.sparse.coo_todense.html#jax.experimental.sparse.coo_todense "jax.experimental.sparse.coo_todense")(mat) | Convert a COO-format sparse matrix to a dense matrix. |
| [`csr_fromdense`](_autosummary/jax.experimental.sparse.csr_fromdense.html#jax.experimental.sparse.csr_fromdense "jax.experimental.sparse.csr_fromdense")(mat, \*\[, nse, index_dtype\]) | Create a CSR-format sparse matrix from a dense matrix. |
| [`csr_matmat`](_autosummary/jax.experimental.sparse.csr_matmat.html#jax.experimental.sparse.csr_matmat "jax.experimental.sparse.csr_matmat")(mat, B, \*\[, transpose\]) | Product of CSR sparse matrix and a dense matrix. |
| [`csr_matvec`](_autosummary/jax.experimental.sparse.csr_matvec.html#jax.experimental.sparse.csr_matvec "jax.experimental.sparse.csr_matvec")(mat, v\[, transpose\]) | Product of CSR sparse matrix and a dense vector. |
| [`csr_todense`](_autosummary/jax.experimental.sparse.csr_todense.html#jax.experimental.sparse.csr_todense "jax.experimental.sparse.csr_todense")(mat) | Convert a CSR-format sparse matrix to a dense matrix. |

### `jax.experimental.sparse.linalg`[\#](#module-jax.experimental.sparse.linalg "Link to this heading")

Sparse linear algebra routines.

|  |  |
|----|----|
| [`spsolve`](_autosummary/jax.experimental.sparse.linalg.spsolve.html#jax.experimental.sparse.linalg.spsolve "jax.experimental.sparse.linalg.spsolve")(data, indices, indptr, b\[, tol, reorder\]) | A sparse direct solver using QR factorization. |
| [`lobpcg_standard`](_autosummary/jax.experimental.sparse.linalg.lobpcg_standard.html#jax.experimental.sparse.linalg.lobpcg_standard "jax.experimental.sparse.linalg.lobpcg_standard")(A, X\[, m, tol\]) | Compute the top-k standard eigenvalues using the LOBPCG routine. |

[](_autosummary/jax.experimental.serialize_executable.deserialize_and_load.html "previous page")

previous

jax.experimental.serialize_executable.deserialize_and_load

[](_autosummary/jax.experimental.sparse.sparsify.html "next page")

next

jax.experimental.sparse.sparsify

Contents

- [Batched-coordinate (BCOO) sparse matrices](#batched-coordinate-bcoo-sparse-matrices)
- [Sparsify transform](#sparsify-transform)
- [Example: sparse logistic regression](#example-sparse-logistic-regression)
- [Sparse API Reference](#sparse-api-reference)
  - [BCOO Data Structure](#bcoo-data-structure)
  - [BCSR Data Structure](#bcsr-data-structure)
  - [Other Sparse Data Structures](#other-sparse-data-structures)
  - [`jax.experimental.sparse.linalg`](#module-jax.experimental.sparse.linalg)

By The JAX authors

© Copyright 2024, The JAX Authors.\
