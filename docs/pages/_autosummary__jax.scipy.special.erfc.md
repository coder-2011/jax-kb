- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.special.erfc

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.special.erfc.rst "Download source file")
-  .pdf

# jax.scipy.special.erfc

## Contents

- [`erfc()`](#jax.scipy.special.erfc)

# jax.scipy.special.erfc[\#](#jax-scipy-special-erfc "Link to this heading")

jax.scipy.special.erfc(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/special.py#L571-L599)[\#](#jax.scipy.special.erfc "Link to this definition")  
The complement of the error function

JAX implementation of [`scipy.special.erfc`](https://scipy.github.io/devdocs/reference/generated/scipy.special.erfc.html#scipy.special.erfc "(in SciPy v1.19.0.dev)").

\\\mathrm{erfc}(x) = \frac{2}{\sqrt\pi} \int\_{x}^\infty e^{-t^2} \mathrm{d}t\\

This is the complement of the error function [`erf()`](jax.scipy.special.erf.html#jax.scipy.special.erf "jax.scipy.special.erf"), `erfc(x)`` ``=`` ``1`` ``-`` ``erf(x)`.

Parameters:  
**x** (*ArrayLike*) – arraylike, real-valued.

Returns:  
array containing values of the complement of the error function.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Notes

The JAX version only supports real-valued inputs.

See also

- [`jax.scipy.special.erf()`](jax.scipy.special.erf.html#jax.scipy.special.erf "jax.scipy.special.erf")

- [`jax.scipy.special.erfcx()`](jax.scipy.special.erfcx.html#jax.scipy.special.erfcx "jax.scipy.special.erfcx")

- [`jax.scipy.special.erfinv()`](jax.scipy.special.erfinv.html#jax.scipy.special.erfinv "jax.scipy.special.erfinv")

[](jax.scipy.special.erf.html "previous page")

previous

jax.scipy.special.erf

[](jax.scipy.special.erfcx.html "next page")

next

jax.scipy.special.erfcx

Contents

- [`erfc()`](#jax.scipy.special.erfc)

By The JAX authors

© Copyright 2024, The JAX Authors.\
