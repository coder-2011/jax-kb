- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.linalg.lu

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.linalg.lu.rst "Download source file")
-  .pdf

# jax.lax.linalg.lu

## Contents

- [`lu()`](#jax.lax.linalg.lu)

# jax.lax.linalg.lu[\#](#jax-lax-linalg-lu "Link to this heading")

jax.lax.linalg.lu(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/linalg.py#L330-L361)[\#](#jax.lax.linalg.lu "Link to this definition")  
LU decomposition with partial pivoting.

Computes the matrix decomposition:

\\P \\ A = L \\ U\\

where \\P\\ is a permutation of the rows of \\A\\, \\L\\ is a lower-triangular matrix with unit-diagonal elements, and \\U\\ is an upper-triangular matrix.

Parameters:  
**x** (*ArrayLike*) – A batch of matrices with shape `[...,`` ``m,`` ``n]`.

Returns:  
A tuple `(lu,`` ``pivots,`` ``permutation)`.

`lu` is a batch of matrices with the same shape and dtype as `x` containing the \\L\\ matrix in its lower triangle and the \\U\\ matrix in its upper triangle. The (unit) diagonal elements of \\L\\ are not represented explicitly.

`pivots` is an int32 array with shape `[...,`` ``min(m,`` ``n)]` representing a sequence of row swaps that should be performed on \\A\\.

`permutation` is an alternative representation of the sequence of row swaps as a permutation, represented as an int32 array with shape `[...,`` ``m]`.

Return type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\]

[](jax.lax.linalg.householder_product.html "previous page")

previous

jax.lax.linalg.householder_product

[](jax.lax.linalg.lu_pivots_to_permutation.html "next page")

next

jax.lax.linalg.lu_pivots_to_permutation

Contents

- [`lu()`](#jax.lax.linalg.lu)

By The JAX authors

© Copyright 2024, The JAX Authors.\
