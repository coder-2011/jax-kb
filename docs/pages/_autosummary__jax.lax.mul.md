- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.mul

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.mul.rst "Download source file")
-  .pdf

# jax.lax.mul

## Contents

- [`mul()`](#jax.lax.mul)

# jax.lax.mul[\#](#jax-lax-mul "Link to this heading")

jax.lax.mul(*x*, *y*, *\**, *out_dtype=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L1190-L1219)[\#](#jax.lax.mul "Link to this definition")  
Elementwise multiplication: \\x \times y\\.

This function lowers directly to the [stablehlo.multiply](https://openxla.org/stablehlo/spec#multiply) operation.

Parameters:  
- **x** (*ArrayLike*) – Input arrays. Must have matching numerical dtypes. If neither is a scalar, `x` and `y` must have the same number of dimensions and be broadcast compatible.

- **y** (*ArrayLike*) – Input arrays. Must have matching numerical dtypes. If neither is a scalar, `x` and `y` must have the same number of dimensions and be broadcast compatible.

- **out_dtype** (*DTypeLike* *\|* *None*) – Optional. Either `None` (default), or a dtype. If it is a dtype, the output will be of the specified dtype. Typically, this is accomplished by casting the inputs to the specified dtype before the multiplication is performed, but on some backends this may be done via a custom kernel.

Returns:  
An array of the same dtype as `x` and `y` containing the product of each pair of broadcasted entries.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.multiply()`](jax.numpy.multiply.html#jax.numpy.multiply "jax.numpy.multiply"): NumPy-style multiplication supporting inputs with mixed dtypes and ranks.

[](jax.lax.min.html "previous page")

previous

jax.lax.min

[](jax.lax.mulhi.html "next page")

next

jax.lax.mulhi

Contents

- [`mul()`](#jax.lax.mul)

By The JAX authors

© Copyright 2024, The JAX Authors.\
