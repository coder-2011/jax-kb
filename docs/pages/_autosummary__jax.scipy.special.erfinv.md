- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.special.erfinv

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.special.erfinv.rst "Download source file")
-  .pdf

# jax.scipy.special.erfinv

## Contents

- [`erfinv()`](#jax.scipy.special.erfinv)

# jax.scipy.special.erfinv[\#](#jax-scipy-special-erfinv "Link to this heading")

jax.scipy.special.erfinv(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/special.py#L601-L623)[\#](#jax.scipy.special.erfinv "Link to this definition")  
The inverse of the error function

JAX implementation of [`scipy.special.erfinv`](https://scipy.github.io/devdocs/reference/generated/scipy.special.erfinv.html#scipy.special.erfinv "(in SciPy v1.19.0.dev)").

Returns the inverse of [`erf()`](jax.scipy.special.erf.html#jax.scipy.special.erf "jax.scipy.special.erf").

Parameters:  
**x** (*ArrayLike*) – arraylike, real-valued.

Returns:  
array containing values of the inverse error function.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Notes

The JAX version only supports real-valued inputs.

See also

- [`jax.scipy.special.erf()`](jax.scipy.special.erf.html#jax.scipy.special.erf "jax.scipy.special.erf")

- [`jax.scipy.special.erfc()`](jax.scipy.special.erfc.html#jax.scipy.special.erfc "jax.scipy.special.erfc")

[](jax.scipy.special.erfcx.html "previous page")

previous

jax.scipy.special.erfcx

[](jax.scipy.special.exp1.html "next page")

next

jax.scipy.special.exp1

Contents

- [`erfinv()`](#jax.scipy.special.erfinv)

By The JAX authors

© Copyright 2024, The JAX Authors.\
