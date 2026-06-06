- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.special.sici

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.special.sici.rst "Download source file")
-  .pdf

# jax.scipy.special.sici

## Contents

- [`sici`](#jax.scipy.special.sici)

# jax.scipy.special.sici[\#](#jax-scipy-special-sici "Link to this heading")

jax.scipy.special.sici *= \<jax.\_src.custom_derivatives.custom_jvp object\>*[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/special.py#L2617-L2669)[\#](#jax.scipy.special.sici "Link to this definition")  
Sine and cosine integrals.

JAX implementation of [`scipy.special.sici`](https://scipy.github.io/devdocs/reference/generated/scipy.special.sici.html#scipy.special.sici "(in SciPy v1.19.0.dev)").

\\\mathrm{Si}(x) = \int_0^x \frac{\sin t}{t} \\ dt\\

\\\mathrm{Ci}(x) = \gamma + \ln(x) + \int_0^x \frac{\cos t - 1}{t} \\ dt\\

where \\\gamma\\ is the Euler–Mascheroni constant.

Parameters:  
**x** (*ArrayLike*) – array-like, real-valued input.

Returns:  
- The first array contains the sine integral values Si(x).

- The second array contains the cosine integral values Ci(x).

Return type:  
A tuple of two arrays, each with the same shape as x

See also

- [`jax.numpy.sinc()`](jax.numpy.sinc.html#jax.numpy.sinc "jax.numpy.sinc")

[](jax.scipy.special.rel_entr.html "previous page")

previous

jax.scipy.special.rel_entr

[](jax.scipy.special.softmax.html "next page")

next

jax.scipy.special.softmax

Contents

- [`sici`](#jax.scipy.special.sici)

By The JAX authors

© Copyright 2024, The JAX Authors.\
