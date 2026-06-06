- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.linalg.hessenberg

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.linalg.hessenberg.rst "Download source file")
-  .pdf

# jax.lax.linalg.hessenberg

## Contents

- [`hessenberg()`](#jax.lax.linalg.hessenberg)

# jax.lax.linalg.hessenberg[\#](#jax-lax-linalg-hessenberg "Link to this heading")

jax.lax.linalg.hessenberg(*a*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/linalg.py#L295-L311)[\#](#jax.lax.linalg.hessenberg "Link to this definition")  
Reduces a square matrix to upper Hessenberg form.

Currently implemented on CPU only.

Parameters:  
**a** (*ArrayLike*) – A floating point or complex square matrix or batch of matrices.

Returns:  
A `(a,`` ``taus)` pair, where the upper triangle and first subdiagonal of `a` contain the upper Hessenberg matrix, and the elements below the first subdiagonal contain the Householder reflectors. For each Householder reflector `taus` contains the scalar factors of the elementary Householder reflectors.

Return type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\]

[](jax.lax.linalg.eigh.html "previous page")

previous

jax.lax.linalg.eigh

[](jax.lax.linalg.householder_product.html "next page")

next

jax.lax.linalg.householder_product

Contents

- [`hessenberg()`](#jax.lax.linalg.hessenberg)

By The JAX authors

© Copyright 2024, The JAX Authors.\
