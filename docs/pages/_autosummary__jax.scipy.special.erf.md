- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.special.erf

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.special.erf.rst "Download source file")
-  .pdf

# jax.scipy.special.erf

## Contents

- [`erf()`](#jax.scipy.special.erf)

# jax.scipy.special.erf[\#](#jax-scipy-special-erf "Link to this heading")

jax.scipy.special.erf(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/special.py#L544-L569)[\#](#jax.scipy.special.erf "Link to this definition")  
The error function

JAX implementation of [`scipy.special.erf`](https://scipy.github.io/devdocs/reference/generated/scipy.special.erf.html#scipy.special.erf "(in SciPy v1.19.0.dev)").

\\\mathrm{erf}(x) = \frac{2}{\sqrt\pi} \int\_{0}^x e^{-t^2} \mathrm{d}t\\

Parameters:  
**x** (*ArrayLike*) – arraylike, real-valued.

Returns:  
array containing values of the error function.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Notes

The JAX version only supports real-valued inputs.

See also

- [`jax.scipy.special.erfc()`](jax.scipy.special.erfc.html#jax.scipy.special.erfc "jax.scipy.special.erfc")

- [`jax.scipy.special.erfcx()`](jax.scipy.special.erfcx.html#jax.scipy.special.erfcx "jax.scipy.special.erfcx")

- [`jax.scipy.special.erfinv()`](jax.scipy.special.erfinv.html#jax.scipy.special.erfinv "jax.scipy.special.erfinv")

[](jax.scipy.special.entr.html "previous page")

previous

jax.scipy.special.entr

[](jax.scipy.special.erfc.html "next page")

next

jax.scipy.special.erfc

Contents

- [`erf()`](#jax.scipy.special.erf)

By The JAX authors

© Copyright 2024, The JAX Authors.\
