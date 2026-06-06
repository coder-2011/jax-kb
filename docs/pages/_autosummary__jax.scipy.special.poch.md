- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.special.poch

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.special.poch.rst "Download source file")
-  .pdf

# jax.scipy.special.poch

## Contents

- [`poch`](#jax.scipy.special.poch)

# jax.scipy.special.poch[\#](#jax-scipy-special-poch "Link to this heading")

jax.scipy.special.poch *= \<jax.\_src.custom_derivatives.custom_jvp object\>*[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/special.py#L3164-L3191)[\#](#jax.scipy.special.poch "Link to this definition")  
The Pochammer symbol.

JAX implementation of [`scipy.special.poch`](https://scipy.github.io/devdocs/reference/generated/scipy.special.poch.html#scipy.special.poch "(in SciPy v1.19.0.dev)").

\\\mathrm{poch}(z, m) = (z)\_m = \frac{\Gamma(z + m)}{\Gamma(z)}\\

where \\\Gamma(z)\\ is the [`gamma()`](jax.scipy.special.gamma.html#jax.scipy.special.gamma "jax.scipy.special.gamma") function.

Parameters:  
- **z** (*ArrayLike*) – arraylike, real-valued

- **m** (*ArrayLike*) – arraylike, real-valued

Returns:  
array of Pochammer values.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Notes

The JAX version supports only real-valued inputs.

[](jax.scipy.special.owens_t.html "previous page")

previous

jax.scipy.special.owens_t

[](jax.scipy.special.polygamma.html "next page")

next

jax.scipy.special.polygamma

Contents

- [`poch`](#jax.scipy.special.poch)

By The JAX authors

© Copyright 2024, The JAX Authors.\
