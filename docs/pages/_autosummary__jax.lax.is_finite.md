- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.is_finite

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.is_finite.rst "Download source file")
-  .pdf

# jax.lax.is_finite

## Contents

- [`is_finite()`](#jax.lax.is_finite)

# jax.lax.is_finite[\#](#jax-lax-is-finite "Link to this heading")

jax.lax.is_finite(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L430-L450)[\#](#jax.lax.is_finite "Link to this definition")  
Elementwise \\\mathrm{isfinite}\\.

This function lowers directly to the [stablehlo.is_finite](https://openxla.org/stablehlo/spec#is_finite) operation.

Parameters:  
**x** (*ArrayLike*) – input array. Must have floating-point type.

Returns:  
Array of boolean dtype with the same shape as `x`, containing `False` where `x` is \\\pm\infty\\ or \\\mathit{NaN}\\, and `True` otherwise.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.isinf()`](jax.numpy.isinf.html#jax.numpy.isinf "jax.numpy.isinf"): return True where array is infinite.

- [`jax.numpy.isnan()`](jax.numpy.isnan.html#jax.numpy.isnan "jax.numpy.isnan"): return True where array is NaN.

[](jax.lax.iota.html "previous page")

previous

jax.lax.iota

[](jax.lax.le.html "next page")

next

jax.lax.le

Contents

- [`is_finite()`](#jax.lax.is_finite)

By The JAX authors

© Copyright 2024, The JAX Authors.\
