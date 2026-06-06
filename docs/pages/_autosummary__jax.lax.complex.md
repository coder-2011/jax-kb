- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.complex

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.complex.rst "Download source file")
-  .pdf

# jax.lax.complex

## Contents

- [`complex()`](#jax.lax.complex)

# jax.lax.complex[\#](#jax-lax-complex "Link to this heading")

jax.lax.complex(*x*, *y*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L794-L819)[\#](#jax.lax.complex "Link to this definition")  
Elementwise make complex number: \\x + jy\\.

This function lowers directly to the [stablehlo.complex](https://openxla.org/stablehlo/spec#complex) operation.

Parameters:  
- **x** (*ArrayLike*) – input arrays. Must have matching floating-point dtypes. If neither is a scalar, the two arrays must have the same number of dimensions and be broadcast-compatible.

- **y** (*ArrayLike*) – input arrays. Must have matching floating-point dtypes. If neither is a scalar, the two arrays must have the same number of dimensions and be broadcast-compatible.

Returns:  
The complex array with the real part given by `x`, and the imaginary part given by `y`. For inputs of dtype float32 or float64, the result will have dtype complex64 or complex128 respectively.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.lax.real()`](jax.lax.real.html#jax.lax.real "jax.lax.real"): elementwise extract real part.

- [`jax.lax.imag()`](jax.lax.imag.html#jax.lax.imag "jax.lax.imag"): elementwise extract imaginary part.

- [`jax.lax.conj()`](jax.lax.conj.html#jax.lax.conj "jax.lax.conj"): elementwise complex conjugate.

[](jax.lax.collapse.html "previous page")

previous

jax.lax.collapse

[](jax.lax.composite.html "next page")

next

jax.lax.composite

Contents

- [`complex()`](#jax.lax.complex)

By The JAX authors

© Copyright 2024, The JAX Authors.\
