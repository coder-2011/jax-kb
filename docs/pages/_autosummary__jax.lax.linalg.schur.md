- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.linalg.schur

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.linalg.schur.rst "Download source file")
-  .pdf

# jax.lax.linalg.schur

## Contents

- [`schur()`](#jax.lax.linalg.schur)

# jax.lax.linalg.schur[\#](#jax-lax-linalg-schur "Link to this heading")

jax.lax.linalg.schur(*x*, *\**, *compute_schur_vectors=True*, *sort_eig_vals=False*, *select_callable=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/linalg.py#L449-L483)[\#](#jax.lax.linalg.schur "Link to this definition")  
Schur decomposition.

Only implemented on CPU.

Computes the Schur decomposition:

\\A = Q \\ U \\ Q^{-H}\\

for a square matrix \\A\\.

Parameters:  
- **x** (*ArrayLike*) – A batch of square matrices with shape `[...,`` ``m,`` ``m]`.

- **compute_schur_vectors** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – If `True`, compute the Schur vectors :\\Q\\, otherwise only \\U\\ is computed.

- **sort_eig_vals** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – Unused.

- **select_callable** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[...,* *Any\]* *\|* *None*) – Unused.

Returns:  
A pair of arrays `U,`` ``Q`, if `compute_schur_vectors=True`, otherwise only `U` is returned.

Return type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\]

[](jax.lax.linalg.qr.html "previous page")

previous

jax.lax.linalg.qr

[](jax.lax.linalg.svd.html "next page")

next

jax.lax.linalg.svd

Contents

- [`schur()`](#jax.lax.linalg.schur)

By The JAX authors

© Copyright 2024, The JAX Authors.\
