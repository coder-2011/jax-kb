- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.linalg.householder_product

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.linalg.householder_product.rst "Download source file")
-  .pdf

# jax.lax.linalg.householder_product

## Contents

- [`householder_product()`](#jax.lax.linalg.householder_product)

# jax.lax.linalg.householder_product[\#](#jax-lax-linalg-householder-product "Link to this heading")

jax.lax.linalg.householder_product(*a*, *taus*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/linalg.py#L313-L328)[\#](#jax.lax.linalg.householder_product "Link to this definition")  
Product of elementary Householder reflectors.

Parameters:  
- **a** (*ArrayLike*) – A matrix with shape `[...,`` ``m,`` ``n]`, whose lower triangle contains elementary Householder reflectors.

- **taus** (*ArrayLike*) – A vector with shape `[...,`` ``k]`, where `k`` ``<`` ``min(m,`` ``n)`, containing the scalar factors of the elementary Householder reflectors.

Returns:  
A batch of orthogonal (unitary) matrices with the same shape as `a`, containing the products of the elementary Householder reflectors.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.lax.linalg.hessenberg.html "previous page")

previous

jax.lax.linalg.hessenberg

[](jax.lax.linalg.lu.html "next page")

next

jax.lax.linalg.lu

Contents

- [`householder_product()`](#jax.lax.linalg.householder_product)

By The JAX authors

© Copyright 2024, The JAX Authors.\
