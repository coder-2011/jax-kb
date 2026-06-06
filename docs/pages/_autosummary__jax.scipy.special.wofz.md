- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.special.wofz

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.special.wofz.rst "Download source file")
-  .pdf

# jax.scipy.special.wofz

## Contents

- [`wofz()`](#jax.scipy.special.wofz)

# jax.scipy.special.wofz[\#](#jax-scipy-special-wofz "Link to this heading")

jax.scipy.special.wofz(*z*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/special.py#L1246-L1266)[\#](#jax.scipy.special.wofz "Link to this definition")  
Faddeeva function.

JAX implementation of [`scipy.special.wofz`](https://scipy.github.io/devdocs/reference/generated/scipy.special.wofz.html#scipy.special.wofz "(in SciPy v1.19.0.dev)").

\\\mathrm{wofz}(z) = e^{-z^2} \mathrm{erfc}(-iz)\\

Parameters:  
**z** (*ArrayLike*) – arraylike, real or complex.

Returns:  
array of complex values of the Faddeeva function.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.scipy.special.erfcx()`](jax.scipy.special.erfcx.html#jax.scipy.special.erfcx "jax.scipy.special.erfcx")

[](jax.scipy.special.sph_harm_y.html "previous page")

previous

jax.scipy.special.sph_harm_y

[](jax.scipy.special.xlog1py.html "next page")

next

jax.scipy.special.xlog1py

Contents

- [`wofz()`](#jax.scipy.special.wofz)

By The JAX authors

© Copyright 2024, The JAX Authors.\
