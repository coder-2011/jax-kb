- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.linalg.eigh

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.linalg.eigh.rst "Download source file")
-  .pdf

# jax.lax.linalg.eigh

## Contents

- [`eigh()`](#jax.lax.linalg.eigh)

# jax.lax.linalg.eigh[\#](#jax-lax-linalg-eigh "Link to this heading")

jax.lax.linalg.eigh(*x*, *\**, *lower=True*, *symmetrize_input=True*, *sort_eigenvalues=True*, *subset_by_index=None*, *implementation=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/linalg.py#L236-L293)[\#](#jax.lax.linalg.eigh "Link to this definition")  
Eigendecomposition of a Hermitian matrix.

Computes the eigenvectors and eigenvalues of a complex Hermitian or real symmetric square matrix.

Parameters:  
- **x** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – A batch of square complex Hermitian or real symmetric matrices with shape `[...,`` ``n,`` ``n]`.

- **lower** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – If `symmetrize_input` is `False`, describes which triangle of the input matrix to use. If `symmetrize_input` is `False`, only the triangle given by `lower` is accessed; the other triangle is ignored and not accessed.

- **symmetrize_input** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – If `True`, the matrix is symmetrized before the eigendecomposition by computing \\\frac{1}{2}(x + x^H)\\.

- **sort_eigenvalues** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – If `True`, the eigenvalues will be sorted in ascending order. If `False` the eigenvalues are returned in an implementation-defined order.

- **subset_by_index** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]* *\|* *None*) – Optional 2-tuple \[start, end\] indicating the range of indices of eigenvalues to compute. For example, is `range_select` = \[n-2,n\], then `eigh` computes the two largest eigenvalues and their eigenvectors.

- **implementation** ([*EighImplementation*](../jax.lax.html#jax.lax.linalg.EighImplementation "jax.lax.linalg.EighImplementation") *\|* *None*) – Optional implementation selection. `QR` uses QR-based decomposition (default for CPU/GPU). `JACOBI` uses Jacobi iteration (GPU/TPU only). `QDWH` uses QDWH spectral divide-and-conquer (default on TPU, TPU only).

Returns:  
A tuple `(v,`` ``w)`.

`v` is an array with the same dtype as `x` such that `v[...,`` ``:,`` ``i]` is the normalized eigenvector corresponding to eigenvalue `w[...,`` ``i]`.

`w` is an array with the same dtype as `x` (or its real counterpart if complex) with shape `[...,`` ``d]` containing the eigenvalues of `x` in ascending order(each repeated according to its multiplicity). If `subset_by_index` is `None` then `d` is equal to `n`. Otherwise `d` is equal to `subset_by_index[1]`` ``-`` ``subset_by_index[0]`.

Return type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\]

[](jax.lax.linalg.eig.html "previous page")

previous

jax.lax.linalg.eig

[](jax.lax.linalg.hessenberg.html "next page")

next

jax.lax.linalg.hessenberg

Contents

- [`eigh()`](#jax.lax.linalg.eigh)

By The JAX authors

© Copyright 2024, The JAX Authors.\
