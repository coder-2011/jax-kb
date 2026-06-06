- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.real

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.real.rst "Download source file")
-  .pdf

# jax.lax.real

## Contents

- [`real()`](#jax.lax.real)

# jax.lax.real[\#](#jax-lax-real "Link to this heading")

jax.lax.real(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L750-L771)[\#](#jax.lax.real "Link to this definition")  
Elementwise extract real part: \\\mathrm{Re}(x)\\.

This function lowers directly to the [stablehlo.real](https://openxla.org/stablehlo/spec#real) operation.

Parameters:  
**x** (*ArrayLike*) – input array. Must have complex dtype.

Returns:  
Array of the same shape as `x` containing its real part. Will have dtype float32 if `x.dtype`` ``==`` ``complex64`, or float64 if `x.dtype`` ``==`` ``complex128`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.lax.complex()`](jax.lax.complex.html#jax.lax.complex "jax.lax.complex"): elementwise construct complex number.

- [`jax.lax.imag()`](jax.lax.imag.html#jax.lax.imag "jax.lax.imag"): elementwise extract imaginary part.

- [`jax.lax.conj()`](jax.lax.conj.html#jax.lax.conj "jax.lax.conj"): elementwise complex conjugate.

[](jax.lax.ragged_dot_general.html "previous page")

previous

jax.lax.ragged_dot_general

[](jax.lax.reciprocal.html "next page")

next

jax.lax.reciprocal

Contents

- [`real()`](#jax.lax.real)

By The JAX authors

© Copyright 2024, The JAX Authors.\
