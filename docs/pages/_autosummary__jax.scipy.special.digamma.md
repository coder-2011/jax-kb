- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.special.digamma

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.special.digamma.rst "Download source file")
-  .pdf

# jax.scipy.special.digamma

## Contents

- [`digamma()`](#jax.scipy.special.digamma)

# jax.scipy.special.digamma[\#](#jax-scipy-special-digamma "Link to this heading")

jax.scipy.special.digamma(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/special.py#L464-L490)[\#](#jax.scipy.special.digamma "Link to this definition")  
The digamma function

JAX implementation of [`scipy.special.digamma`](https://scipy.github.io/devdocs/reference/generated/scipy.special.digamma.html#scipy.special.digamma "(in SciPy v1.19.0.dev)").

\\\mathrm{digamma}(z) = \psi(z) = \frac{\mathrm{d}}{\mathrm{d}z}\log \Gamma(z)\\

where \\\Gamma(z)\\ is the [`gamma()`](jax.scipy.special.gamma.html#jax.scipy.special.gamma "jax.scipy.special.gamma") function.

Parameters:  
**x** (*ArrayLike*) – arraylike, real-valued.

Returns:  
array containing values of the digamma function.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Notes

The JAX version of digamma accepts real-valued inputs.

See also

- [`jax.scipy.special.gamma()`](jax.scipy.special.gamma.html#jax.scipy.special.gamma "jax.scipy.special.gamma")

- [`jax.scipy.special.polygamma()`](jax.scipy.special.polygamma.html#jax.scipy.special.polygamma "jax.scipy.special.polygamma")

[](jax.scipy.special.dawsn.html "previous page")

previous

jax.scipy.special.dawsn

[](jax.scipy.special.entr.html "next page")

next

jax.scipy.special.entr

Contents

- [`digamma()`](#jax.scipy.special.digamma)

By The JAX authors

© Copyright 2024, The JAX Authors.\
