- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.add

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.add.rst "Download source file")
-  .pdf

# jax.lax.add

## Contents

- [`add()`](#jax.lax.add)

# jax.lax.add[\#](#jax-lax-add "Link to this heading")

jax.lax.add(*x*, *y*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L1142-L1165)[\#](#jax.lax.add "Link to this definition")  
Elementwise addition: \\x + y\\.

This function lowers directly to the [stablehlo.add](https://openxla.org/stablehlo/spec#add) operation.

Parameters:  
- **x** (*ArrayLike*) – Input arrays. Must have matching numerical dtypes. If neither is a scalar, `x` and `y` must have the same number of dimensions and be broadcast compatible.

- **y** (*ArrayLike*) – Input arrays. Must have matching numerical dtypes. If neither is a scalar, `x` and `y` must have the same number of dimensions and be broadcast compatible.

Returns:  
An array of the same dtype as `x` and `y` containing the sum of each pair of broadcasted entries.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.add()`](jax.numpy.add.html#jax.numpy.add "jax.numpy.add"): NumPy-style addition supporting inputs with mixed dtypes and ranks.

[](jax.lax.acosh.html "previous page")

previous

jax.lax.acosh

[](jax.lax.after_all.html "next page")

next

jax.lax.after_all

Contents

- [`add()`](#jax.lax.add)

By The JAX authors

© Copyright 2024, The JAX Authors.\
