- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.conj

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.conj.rst "Download source file")
-  .pdf

# jax.lax.conj

## Contents

- [`conj()`](#jax.lax.conj)

# jax.lax.conj[\#](#jax-lax-conj "Link to this heading")

jax.lax.conj(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L820-L845)[\#](#jax.lax.conj "Link to this definition")  
Elementwise complex conjugate function: \\\overline{x}\\.

This function lowers to a combination of [stablehlo.real](https://openxla.org/stablehlo/spec#real), [stablehlo.imag](https://openxla.org/stablehlo/spec#imag), and [stablehlo.complex](https://openxla.org/stablehlo/spec#complex).

Parameters:  
**x** (*ArrayLike*) – input array. Must have complex dtype.

Returns:  
Array of the same shape and dtype as `x` containing its complex conjugate.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.lax.complex()`](jax.lax.complex.html#jax.lax.complex "jax.lax.complex"): elementwise construct complex number.

- [`jax.lax.real()`](jax.lax.real.html#jax.lax.real "jax.lax.real"): elementwise extract real part.

- [`jax.lax.imag()`](jax.lax.imag.html#jax.lax.imag "jax.lax.imag"): elementwise extract imaginary part.

- [`jax.lax.abs()`](jax.lax.abs.html#jax.lax.abs "jax.lax.abs"): elementwise absolute value / complex magnitude.

[](jax.lax.concatenate.html "previous page")

previous

jax.lax.concatenate

[](jax.lax.conv.html "next page")

next

jax.lax.conv

Contents

- [`conj()`](#jax.lax.conj)

By The JAX authors

© Copyright 2024, The JAX Authors.\
