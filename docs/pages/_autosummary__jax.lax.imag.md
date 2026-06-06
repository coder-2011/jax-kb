- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.imag

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.imag.rst "Download source file")
-  .pdf

# jax.lax.imag

## Contents

- [`imag()`](#jax.lax.imag)

# jax.lax.imag[\#](#jax-lax-imag "Link to this heading")

jax.lax.imag(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L772-L793)[\#](#jax.lax.imag "Link to this definition")  
Elementwise extract imaginary part: \\\mathrm{Im}(x)\\.

This function lowers directly to the [stablehlo.imag](https://openxla.org/stablehlo/spec#imag) operation.

Parameters:  
**x** (*ArrayLike*) – input array. Must have complex dtype.

Returns:  
Array of the same shape as `x` containing its imaginary part. Will have dtype float32 if `x.dtype`` ``==`` ``complex64`, or float64 if `x.dtype`` ``==`` ``complex128`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.lax.complex()`](jax.lax.complex.html#jax.lax.complex "jax.lax.complex"): elementwise construct complex number.

- [`jax.lax.real()`](jax.lax.real.html#jax.lax.real "jax.lax.real"): elementwise extract real part.

- [`jax.lax.conj()`](jax.lax.conj.html#jax.lax.conj "jax.lax.conj"): elementwise complex conjugate.

[](jax.lax.igammac.html "previous page")

previous

jax.lax.igammac

[](jax.lax.index_in_dim.html "next page")

next

jax.lax.index_in_dim

Contents

- [`imag()`](#jax.lax.imag)

By The JAX authors

© Copyright 2024, The JAX Authors.\
