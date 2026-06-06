- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.special.erfcx

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.special.erfcx.rst "Download source file")
-  .pdf

# jax.scipy.special.erfcx

## Contents

- [`erfcx()`](#jax.scipy.special.erfcx)

# jax.scipy.special.erfcx[\#](#jax-scipy-special-erfcx "Link to this heading")

jax.scipy.special.erfcx(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/special.py#L671-L699)[\#](#jax.scipy.special.erfcx "Link to this definition")  
Scaled complementary error function.

JAX implementation of [`scipy.special.erfcx`](https://scipy.github.io/devdocs/reference/generated/scipy.special.erfcx.html#scipy.special.erfcx "(in SciPy v1.19.0.dev)").

\\\mathrm{erfcx}(x) = e^{x^2} \mathrm{erfc}(x)\\

This is numerically stable for large positive `x`, unlike the naive formula which overflows.

Parameters:  
**x** (*ArrayLike*) – arraylike, real or complex.

Returns:  
array containing values of the scaled complementary error function.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.scipy.special.erfc()`](jax.scipy.special.erfc.html#jax.scipy.special.erfc "jax.scipy.special.erfc")

- [`jax.scipy.special.erf()`](jax.scipy.special.erf.html#jax.scipy.special.erf "jax.scipy.special.erf")

- [`jax.scipy.special.wofz()`](jax.scipy.special.wofz.html#jax.scipy.special.wofz "jax.scipy.special.wofz")

[](jax.scipy.special.erfc.html "previous page")

previous

jax.scipy.special.erfc

[](jax.scipy.special.erfinv.html "next page")

next

jax.scipy.special.erfinv

Contents

- [`erfcx()`](#jax.scipy.special.erfcx)

By The JAX authors

© Copyright 2024, The JAX Authors.\
