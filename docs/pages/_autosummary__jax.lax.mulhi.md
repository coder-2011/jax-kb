- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.mulhi

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.mulhi.rst "Download source file")
-  .pdf

# jax.lax.mulhi

## Contents

- [`mulhi()`](#jax.lax.mulhi)

# jax.lax.mulhi[\#](#jax-lax-mulhi "Link to this heading")

jax.lax.mulhi(*x*, *y*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L1221-L1240)[\#](#jax.lax.mulhi "Link to this definition")  
Elementwise multiply-high: high bits of \\x \times y\\.

For N-bit integer inputs, this function computes the upper N bits of the full 2N-bit product.

Parameters:  
- **x** (*ArrayLike*) – Input arrays. Must have an integer dtype. If neither is a scalar, `x` and `y` must have the same number of dimensions and be broadcast compatible.

- **y** (*ArrayLike*) – Input arrays. Must have an integer dtype. If neither is a scalar, `x` and `y` must have the same number of dimensions and be broadcast compatible.

Returns:  
An array of the same dtype as `x` and `y` containing the most significant N bits of the 2N-bit product of each pair of broadcasted entries.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.lax.mul.html "previous page")

previous

jax.lax.mul

[](jax.lax.ne.html "next page")

next

jax.lax.ne

Contents

- [`mulhi()`](#jax.lax.mulhi)

By The JAX authors

© Copyright 2024, The JAX Authors.\
